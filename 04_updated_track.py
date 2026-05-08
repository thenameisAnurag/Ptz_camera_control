"""
PTZ Camera — Threaded RTSP + GPU YOLO + Adaptive-Zoom Smooth Tracking
=====================================================================
Architecture:
  Thread 1 (CaptureThread)  — reads RTSP frames, always latest, no buffer lag
  Thread 2 (YOLOThread)     — GPU inference, non-blocking
  Thread 3 (PTZThread)      — sends SOAP commands, never blocks main loop
  Main thread               — OpenCV display + keyboard

Keys (click video window first):
  T           → Toggle tracking ON / OFF
  W/S/A/D     → Manual tilt/pan
  Z/X         → Zoom in/out
  SPACE       → Stop
  H           → Home
  Q           → Quit

Run:
  pip install -r requirements.txt
  python 04_updated_track.py
"""

import os
import requests
from requests.auth import HTTPDigestAuth
import urllib3
import xml.etree.ElementTree as ET
import time
import threading
import queue
import numpy as np
import cv2
import torch
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings()

# ── Config ───────────────────────────────────────────────────────────────────
IP        = os.getenv("CAMERA_IP")
USERNAME  = os.getenv("CAMERA_USERNAME")
PASSWORD  = os.getenv("CAMERA_PASSWORD")

if not all([IP, USERNAME, PASSWORD]):
    raise RuntimeError("Missing camera credentials. Copy .env.example → .env and fill in values.")

SOAP_URL  = f"https://{IP}/onvif/device_service"
AUTH      = HTTPDigestAuth(USERNAME, PASSWORD)
HEADERS   = {"Content-Type": "application/soap+xml"}

SPEED     = float(os.getenv("PTZ_SPEED", "0.35"))
ZOOM_SPD  = float(os.getenv("ZOOM_SPEED", "0.3"))
TRACK_SPD = float(os.getenv("TRACK_SPEED", "0.25"))
DEAD_ZONE = float(os.getenv("DEAD_ZONE", "0.10"))   # fraction of frame — no move inside this radius

TRACK_CLASS    = int(os.getenv("TRACK_CLASS", "39"))    # COCO 39 = bottle
YOLO_CONF      = float(os.getenv("YOLO_CONF", "0.20"))  # low threshold — small/far objects
STALE_FRAMES   = int(os.getenv("STALE_FRAMES", "15"))   # clear last_box after this many frames without detection
YOLO_MODEL     = os.getenv("YOLO_MODEL", "yolov8n.pt")

# ── SOAP ─────────────────────────────────────────────────────────────────────
def soap(body: str) -> str:
    envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
  <s:Body>{body}</s:Body>
</s:Envelope>"""
    r = requests.post(SOAP_URL, data=envelope, headers=HEADERS,
                      auth=AUTH, verify=False, timeout=8)
    return r.text

def get_token():
    resp = soap('<GetProfiles xmlns="http://www.onvif.org/ver10/media/wsdl"/>')
    root = ET.fromstring(resp)
    return root.findall('.//{http://www.onvif.org/ver10/media/wsdl}Profiles')[0].attrib['token']

def get_rtsp_url(token):
    body = f"""<GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl">
  <StreamSetup>
    <Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>
    <Transport xmlns="http://www.onvif.org/ver10/schema"><Protocol>RTSP</Protocol></Transport>
  </StreamSetup>
  <ProfileToken>{token}</ProfileToken>
</GetStreamUri>"""
    resp = soap(body)
    uri  = ET.fromstring(resp).find('.//{http://www.onvif.org/ver10/schema}Uri')
    return uri.text.replace("rtsp://", f"rtsp://{USERNAME}:{PASSWORD}@") if uri is not None else None

def _ptz_move(token, pan=0.0, tilt=0.0, zoom=0.0):
    soap(f"""<ContinuousMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
  <Velocity>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema" x="{pan}" y="{tilt}"/>
    <Zoom    xmlns="http://www.onvif.org/ver10/schema" x="{zoom}"/>
  </Velocity>
</ContinuousMove>""")

def _ptz_stop(token):
    soap(f"""<Stop xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
  <PanTilt>true</PanTilt><Zoom>true</Zoom>
</Stop>""")

def _go_home(token):
    resp = soap(f"""<GotoHomePosition xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
  <Speed>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema" x="0.5" y="0.5"/>
    <Zoom    xmlns="http://www.onvif.org/ver10/schema" x="0.5"/>
  </Speed>
</GotoHomePosition>""")
    if "Fault" in resp:
        _ptz_stop(token)

def _set_home(token):
    soap(f"""<SetHomePosition xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
</SetHomePosition>""")

# ── Thread 1: RTSP Capture ────────────────────────────────────────────────────
class CaptureThread(threading.Thread):
    """
    Continuously reads the RTSP stream into a single slot (latest frame only).
    Uses an Event so the main thread wakes immediately when a frame is ready.
    """
    def __init__(self, rtsp_url):
        super().__init__(daemon=True)
        self.rtsp_url   = rtsp_url
        self.frame      = None
        self.lock       = threading.Lock()
        self.running    = True
        self.connected  = False
        self.frame_ready = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connected = cap.isOpened()

        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                self.frame_ready.set()   # wake main thread
            else:
                time.sleep(0.02)

        cap.release()

    def get_frame(self):
        self.frame_ready.wait(timeout=0.1)   # block until frame or timeout
        self.frame_ready.clear()
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.frame_ready.set()  # unblock any waiting get_frame

# ── Thread 2: YOLO Inference ──────────────────────────────────────────────────
class YOLOThread(threading.Thread):
    """
    Pulls frames from an input queue, runs GPU YOLO, pushes results to output queue.
    Non-blocking — main loop never waits for inference.
    """
    def __init__(self, model, input_q, output_q):
        super().__init__(daemon=True)
        self.model    = model
        self.input_q  = input_q
        self.output_q = output_q
        self.running  = True

    def run(self):
        while self.running:
            try:
                frame = self.input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            results  = self.model(frame, classes=[TRACK_CLASS],
                                  conf=YOLO_CONF, verbose=False)
            boxes_out = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes_out.append((int(x1), int(y1), int(x2), int(y2), conf))

            # Keep only latest result
            while not self.output_q.empty():
                try: self.output_q.get_nowait()
                except: pass
            self.output_q.put(boxes_out)

    def stop(self):
        self.running = False

# ── Thread 3: PTZ Command Queue ───────────────────────────────────────────────
class PTZThread(threading.Thread):
    """
    Serializes all PTZ SOAP calls so they never block the display loop.
    Drops queued commands if a newer one arrives (except STOP — always executes).
    """
    def __init__(self, token):
        super().__init__(daemon=True)
        self.token   = token
        self.q       = queue.Queue(maxsize=1)
        self.running = True

    def send(self, cmd: dict):
        # Drop old command if queue is full (keeps latency low)
        while self.q.full():
            try: self.q.get_nowait()
            except: pass
        self.q.put(cmd)

    def run(self):
        while self.running:
            try:
                cmd = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            action = cmd.get('action')
            if action == 'move':
                _ptz_move(self.token, cmd['pan'], cmd['tilt'], cmd.get('zoom', 0.0))
            elif action == 'stop':
                _ptz_stop(self.token)
            elif action == 'home':
                _go_home(self.token)

    def stop(self):
        self.running = False

# ── Tracking helpers ──────────────────────────────────────────────────────────
def best_box(boxes):
    """Return largest box (highest area = closest bottle)."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

# Adaptive zoom + proportional speed config
ZOOM_TARGET_MIN  = 0.012  # object is "far"   if box < 1.2% of frame area
ZOOM_TARGET_MAX  = 0.12   # object is "close"  if box > 12% of frame area
ZOOM_DEAD_ZONE   = 0.30   # +/- 30% tolerance around target — no zoom in this band
                          # e.g. target=1.2%, dead zone = 0.84%–1.56% → no zoom
ZOOM_IN_SPD      = 0.12   # max zoom in speed  (gentle — not a sudden jump)
ZOOM_OUT_SPD     = -0.08  # max zoom out speed (gentle)
ZOOM_INTERVAL    = 1.2    # seconds between zoom adjustments (let zoom settle first)
TRACK_SPD_MIN    = 0.08   # min pan/tilt speed (just outside dead zone)
TRACK_SPD_MAX    = 0.35   # max pan/tilt speed (object at frame edge)
PAN_DIR          = -1     # flip to +1 if pan direction is wrong
TILT_DIR         = +1     # flip to -1 if tilt direction is wrong

def offset_to_ptz(cx_obj, cy_obj, w, h):
    """
    Proportional pan/tilt — speed scales with distance from center.
    Dead zone at center prevents jitter when object is nearly centered.
    Sign convention verified against this camera (PAN_DIR / TILT_DIR above).
    """
    dx = (cx_obj - w / 2) / (w / 2)   # -1 (left) to +1 (right)
    dy = (cy_obj - h / 2) / (h / 2)   # -1 (top)  to +1 (bottom)

    if abs(dx) > DEAD_ZONE:
        # Scale speed linearly: TRACK_SPD_MIN at edge of dead zone → TRACK_SPD_MAX at frame edge
        magnitude = (abs(dx) - DEAD_ZONE) / (1.0 - DEAD_ZONE)
        speed = TRACK_SPD_MIN + magnitude * (TRACK_SPD_MAX - TRACK_SPD_MIN)
        pan = PAN_DIR * speed * np.sign(dx)
    else:
        pan = 0.0

    if abs(dy) > DEAD_ZONE:
        magnitude = (abs(dy) - DEAD_ZONE) / (1.0 - DEAD_ZONE)
        speed = TRACK_SPD_MIN + magnitude * (TRACK_SPD_MAX - TRACK_SPD_MIN)
        tilt = TILT_DIR * speed * np.sign(dy)
    else:
        tilt = 0.0

    return float(pan), float(tilt)

def compute_zoom(box_area_ratio, last_zoom_time):
    """
    Zoom only when object is significantly outside the target size band.
    Dead zone prevents constant zooming — only acts when clearly too far/close.
    Cooldown (ZOOM_INTERVAL) lets the zoom physically settle before next adjustment.

    Returns (zoom_velocity, should_send):
      zoom_velocity — value to send to PTZ (0 = stop zoom)
      should_send   — True only when a new zoom command should be issued
    """
    now = time.time()
    elapsed = now - last_zoom_time

    # Target center and dead zone band
    target_min_with_dz = ZOOM_TARGET_MIN * (1.0 - ZOOM_DEAD_ZONE)
    target_max_with_dz = ZOOM_TARGET_MAX * (1.0 + ZOOM_DEAD_ZONE)

    if box_area_ratio < target_min_with_dz:
        # Clearly too far — zoom in
        if elapsed < ZOOM_INTERVAL:
            return 0.0, False   # still settling from last zoom
        # Proportional: zoom faster when very far
        deficit = min((target_min_with_dz - box_area_ratio) / target_min_with_dz, 1.0)
        spd = ZOOM_IN_SPD * (0.4 + 0.6 * deficit)   # 40-100% of max speed
        return float(spd), True

    elif box_area_ratio > target_max_with_dz:
        # Clearly too close — zoom out
        if elapsed < ZOOM_INTERVAL:
            return 0.0, False
        excess = min((box_area_ratio - target_max_with_dz) / target_max_with_dz, 1.0)
        spd = ZOOM_OUT_SPD * (0.4 + 0.6 * excess)
        return float(spd), True

    else:
        # Inside dead zone — no zoom needed
        return 0.0, False

# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(frame, status, mode, bottle_found, fps):
    h, w = frame.shape[:2]

    def bar(y0, y1):
        b = frame.copy()
        cv2.rectangle(b, (0, y0), (w, y1), (0, 0, 0), -1)
        cv2.addWeighted(b, 0.55, frame, 0.45, 0, frame)

    bar(0, 44); bar(h - 40, h)

    # Mode badge
    mc = (0, 200, 255) if mode == "TRACK" else (0, 230, 80)
    cv2.rectangle(frame, (8, 6), (115, 38), mc, -1)
    cv2.putText(frame, mode, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    # Status
    cv2.putText(frame, status, (125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2)

    # FPS
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

    # Bottle indicator (track mode)
    if mode == "TRACK":
        dc = (0, 255, 0) if bottle_found else (0, 0, 255)
        dl = "LOCKED" if bottle_found else "SEARCHING"
        cv2.circle(frame, (w - 200, 22), 8, dc, -1)
        cv2.putText(frame, dl, (w - 185, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dc, 2)

    # LIVE
    cv2.circle(frame, (w - 22, 22), 8, (0, 0, 220), -1)
    cv2.putText(frame, "LIVE", (w-60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,220), 2)

    # Crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx-25, cy), (cx+25, cy), (255,255,255), 1)
    cv2.line(frame, (cx, cy-25), (cx, cy+25), (255,255,255), 1)
    cv2.circle(frame, (cx, cy), int(DEAD_ZONE * w / 2), (100,100,255), 1)

    # Controls
    ctrl = ("T:Manual  SPACE:Stop  H:Home  Q:Quit" if mode == "TRACK"
            else "W/S:Tilt  A/D:Pan  Z/X:Zoom  T:Track  SPACE:Stop  H:Home  Q:Quit")
    cv2.putText(frame, ctrl, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)

    return frame

# ── Key map ───────────────────────────────────────────────────────────────────
KEY_MAP = {
    ord('w'): ("TILT UP",   dict(action='move', pan=0,      tilt= SPEED, zoom=0)),
    ord('s'): ("TILT DOWN", dict(action='move', pan=0,      tilt=-SPEED, zoom=0)),
    ord('a'): ("PAN LEFT",  dict(action='move', pan=-SPEED, tilt=0,      zoom=0)),
    ord('d'): ("PAN RIGHT", dict(action='move', pan= SPEED, tilt=0,      zoom=0)),
    ord('z'): ("ZOOM IN",   dict(action='move', pan=0,      tilt=0,      zoom= ZOOM_SPD)),
    ord('x'): ("ZOOM OUT",  dict(action='move', pan=0,      tilt=0,      zoom=-ZOOM_SPD)),
}

# ── Phase 1: Auto test ────────────────────────────────────────────────────────
def run_auto_test(token):
    print("\n" + "="*50)
    print("  PHASE 1 — Auto Test  (Ctrl+C to skip)")
    print("="*50)
    steps = [("LEFT", -SPEED, 0), ("RIGHT", SPEED, 0),
             ("UP", 0, SPEED),    ("DOWN", 0, -SPEED)]
    try:
        for label, pan, tilt in steps:
            print(f"  ▶ {label} 5s...")
            _ptz_move(token, pan=pan, tilt=tilt)
            time.sleep(5)
            _ptz_stop(token)
            time.sleep(1)
        print("  🏠 Home...")
        _go_home(token)
        time.sleep(4)
        print("✅ Auto test done\n")
    except KeyboardInterrupt:
        _ptz_stop(token)
        print("\n⏭️  Skipped\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # YOLO + GPU
    print("📦 Loading YOLO...")
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    model = YOLO(YOLO_MODEL)
    model.to(device)
    # Warm up
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    print("✅ YOLO ready\n")

    # Camera
    print("🔌 Connecting to camera...")
    token = get_token()
    print(f"✅ Profile: {token}")
    _set_home(token)

    rtsp_url = get_rtsp_url(token)
    print(f"📹 RTSP stream acquired\n")

    # Phase 1
    run_auto_test(token)

    # Start threads
    yolo_in_q  = queue.Queue(maxsize=1)
    yolo_out_q = queue.Queue(maxsize=1)

    cap_thread  = CaptureThread(rtsp_url)
    yolo_thread = YOLOThread(model, yolo_in_q, yolo_out_q)
    ptz_thread  = PTZThread(token)

    cap_thread.start()
    yolo_thread.start()
    ptz_thread.start()

    time.sleep(1.5)   # let RTSP buffer fill

    WIN = "PTZ Camera  |  T=Track  Q=Quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    # State
    track_mode     = False
    current_action = None
    status         = "READY"
    last_box_data  = None    # (x1,y1,x2,y2,conf) — only updated when fresh detection
    stale_counter  = 0
    last_track_cmd = (0.0, 0.0, 0.0)
    last_zoom_time = 0.0    # timestamp of last zoom command sent
    frame_idx      = 0
    fps_t          = time.time()
    fps            = 0.0

    print("Phase 2 — Live feed active. Click the window, then press keys.\n")

    while True:
        frame = cap_thread.get_frame()
        if frame is None:
            continue

        # Flip (camera upside-down)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w  = frame.shape[:2]
        frame_idx += 1

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_t, 1e-5))
        fps_t = now

        # ── Feed frame to YOLO (non-blocking, drop if busy) ───────────────
        if track_mode:
            if not yolo_in_q.full():
                yolo_in_q.put(frame.copy())

            # Collect latest YOLO result (non-blocking)
            try:
                boxes = yolo_out_q.get_nowait()
                box   = best_box(boxes)
                if box:
                    last_box_data = box
                    stale_counter = 0
                else:
                    stale_counter += 1
                    if stale_counter >= STALE_FRAMES:
                        last_box_data = None
            except queue.Empty:
                pass

        # ── Tracking PTZ control ──────────────────────────────────────────
        if track_mode:
            if last_box_data:
                x1, y1, x2, y2, conf = last_box_data
                cx_obj = (x1 + x2) // 2
                cy_obj = (y1 + y2) // 2

                # Draw box + line to crosshair
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"BOTTLE {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.line(frame, (w//2, h//2), (cx_obj, cy_obj), (0, 255, 255), 1)

                pan, tilt = offset_to_ptz(cx_obj, cy_obj, w, h)

                # Adaptive zoom — only fires when outside dead zone AND cooldown elapsed
                box_area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                zoom, should_zoom = compute_zoom(box_area_ratio, last_zoom_time)

                # Draw size indicator next to box
                pct = box_area_ratio * 100
                if zoom > 0:
                    z_txt = f"ZOOM IN  {pct:.1f}%"
                    z_col = (0, 255, 255)
                elif zoom < 0:
                    z_txt = f"ZOOM OUT {pct:.1f}%"
                    z_col = (0, 165, 255)
                else:
                    z_txt = f"OK {pct:.1f}%"
                    z_col = (100, 255, 100)
                cv2.putText(frame, z_txt, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, z_col, 1)

                # Pan/tilt command — send every frame if direction changed
                pt_changed = (pan, tilt) != last_track_cmd[:2]
                if pt_changed or (should_zoom and zoom != last_track_cmd[2]):
                    last_track_cmd = (pan, tilt, zoom if should_zoom else last_track_cmd[2])

                    if pan == 0.0 and tilt == 0.0 and (not should_zoom or zoom == 0.0):
                        ptz_thread.send({'action': 'stop'})
                        status = "CENTERED ✓"
                    else:
                        effective_zoom = zoom if should_zoom else 0.0
                        if should_zoom and zoom != 0.0:
                            last_zoom_time = time.time()   # start cooldown
                        dirs = []
                        if cy_obj < h // 2: dirs.append("↑")
                        if cy_obj > h // 2: dirs.append("↓")
                        if cx_obj < w // 2: dirs.append("←")
                        if cx_obj > w // 2: dirs.append("→")
                        if effective_zoom > 0: dirs.append("Z+")
                        if effective_zoom < 0: dirs.append("Z-")
                        status = "TRACK " + " ".join(dirs)
                        ptz_thread.send({'action': 'move', 'pan': pan,
                                         'tilt': tilt, 'zoom': effective_zoom})

            else:
                # No target — stop if we were moving
                if last_track_cmd != (0.0, 0.0, 0.0):
                    last_track_cmd = (0.0, 0.0, 0.0)
                    ptz_thread.send({'action': 'stop'})
                status = "SEARCHING..."

        # ── HUD ──────────────────────────────────────────────────────────
        bottle_found = (last_box_data is not None) and track_mode
        draw_hud(frame, status, "TRACK" if track_mode else "MANUAL", bottle_found, fps)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ─────────────────────────────────────────────────
        if key == ord('q'):
            print("👋 Quit")
            ptz_thread.send({'action': 'stop'})
            break

        elif key == ord('t'):
            track_mode     = not track_mode
            # *** Reset ALL stale state when toggling ***
            last_box_data  = None
            stale_counter  = 0
            last_track_cmd = (0.0, 0.0, 0.0)
            current_action = None
            # Clear YOLO queues
            while not yolo_in_q.empty():
                try: yolo_in_q.get_nowait()
                except: pass
            while not yolo_out_q.empty():
                try: yolo_out_q.get_nowait()
                except: pass
            ptz_thread.send({'action': 'stop'})   # always stop on toggle
            if track_mode:
                status = "SEARCHING..."
                print("🎯 Tracking ON")
            else:
                status = "MANUAL"
                print("🕹️  Manual mode")

        elif key == ord('h'):
            track_mode     = False
            last_box_data  = None
            last_track_cmd = (0.0, 0.0, 0.0)
            current_action = 'home'
            status         = "HOME"
            ptz_thread.send({'action': 'home'})

        elif key == ord(' '):
            track_mode     = False
            last_box_data  = None
            last_track_cmd = (0.0, 0.0, 0.0)
            current_action = 'stop'
            status         = "STOPPED"
            ptz_thread.send({'action': 'stop'})

        elif key in KEY_MAP and not track_mode:
            label, cmd = KEY_MAP[key]
            if current_action != key:
                current_action = key
                status         = label
                ptz_thread.send(cmd)

        elif key != 0xFF and key not in KEY_MAP and not track_mode:
            # A key was pressed but it's not a movement key — stop
            if current_action not in (None, 'stop', 'home'):
                current_action = None
                status         = "STOPPED"
                ptz_thread.send({'action': 'stop'})

        elif key == 0xFF and not track_mode:
            # No key held — release movement
            if current_action not in (None, 'stop', 'home'):
                current_action = None
                status         = "STOPPED"
                ptz_thread.send({'action': 'stop'})

    # Cleanup
    cap_thread.stop()
    yolo_thread.stop()
    ptz_thread.stop()
    cv2.destroyAllWindows()
    print("✅ Done")

if __name__ == "__main__":
    main()
