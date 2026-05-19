"""
PTZ Camera — Threaded RTSP + GPU YOLO + Adaptive-Zoom Smooth Tracking
=====================================================================
Architecture:
  Thread 1 (CaptureThread)  — reads RTSP frames, always latest, no buffer lag
  Thread 2 (YOLOThread)     — GPU inference + BoT-SORT, non-blocking
  Thread 3 (PTZThread)      — sends SOAP commands, never blocks main loop
  Main thread               — OpenCV display + keyboard

Keys (click video window first):
  T           → Toggle tracking ON / OFF
  W/S/A/D     → Manual tilt/pan
  Z/X         → Zoom in/out
  SPACE       → Stop
  H           → ONVIF Home (camera-stored)
  G           → Go to custom Home (from home.json — runway position)
  P           → Print current PTZ position (use to fill home.json)
  R           → Capture current PTZ as new custom Home (saves home.json + .bak)
  I           → Toggle ROI mode (then drag-mouse on frame to set ROI)
  C           → Clear ROI
  Q           → Quit

Automatic behaviors:
  - If a tracked target is lost for LOST_HOME_SECONDS (default 10s),
    the camera automatically returns to ONVIF home (or custom home
    if LOST_HOME_TARGET=custom is set). Tracking remains enabled.

Run:
  pip install -r requirements.txt
  python track.py
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

# Feature 1 — Manual home position (Task 1 of TDD roadmap)
from home_position import HomePosition, HomeConfigError, backup_then_save
from track_helpers import (
    absolute_move,
    get_current_position,
    goto_custom_home,
)

# Feature 2 — ROI confinement (Task 2)
from roi import ROI, ROIError, filter_boxes_in_roi

# Feature 4 — ID-aware tracking via BoT-SORT (Task 4)
# Pure-logic module — does the ID commit / coast / re-acquire policy.
# BoT-SORT itself lives inside Ultralytics; we just feed its output here.
from tracker_state import TrackerState, pick_track

# Feature 5 — Auto return-to-home on persistent target loss (Task 5)
# Wall-clock timer: if no confirmed lock for N seconds, send a single
# home command. See lost_home_timer.py for the state machine.
from lost_home_timer import LostHomeTimer

# Feature 6 — Smooth steering control law (Task 6)
# Quadratic speed curve replaces the old linear ramp. Camera now creeps
# slowly when target is mid-frame, only ramps up near frame edges.
# This fixes the left-right oscillation observed in field tests.
from steering import compute_steering, SteeringConfig

# TensorRT .engine support — when using .engine files, model.track() is
# unavailable, so we need to manually instantiate the tracker.
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace
import yaml

load_dotenv()
urllib3.disable_warnings()

# ── Config ───────────────────────────────────────────────────────────────────
IP        = "192.168.0.88"
USERNAME  = "admin"
PASSWORD  = "qwerty12@"

if not all([IP, USERNAME, PASSWORD]):
    raise RuntimeError("Missing camera credentials. Copy .env.example → .env and fill in values.")

SOAP_URL  = f"https://{IP}/onvif/device_service"
AUTH      = HTTPDigestAuth(USERNAME, PASSWORD)
HEADERS   = {"Content-Type": "application/soap+xml"}

SPEED     = float(os.getenv("PTZ_SPEED", "0.50"))
ZOOM_SPD  = float(os.getenv("ZOOM_SPEED", "0.20"))
TRACK_SPD = float(os.getenv("TRACK_SPEED", "0.35"))
DEAD_ZONE = float(os.getenv("DEAD_ZONE", "0.05"))   # fraction of frame — no move inside this radius. Widened from 0.10 to reduce over-correction near center.

TRACK_CLASS    = int(os.getenv("TRACK_CLASS", "0"))    # COCO 39 = bottle
YOLO_CONF      = float(os.getenv("YOLO_CONF", "0.20"))  # low threshold — small/far objects
STALE_FRAMES   = int(os.getenv("STALE_FRAMES", "15"))   # clear last_box after this many frames without detection
# yolo11s gives roughly 2-3x the throughput of yolo11m on the same hardware
# while keeping accuracy decent for medium-to-large objects (bottle in hand,
# airplane on approach). If accuracy regresses on far objects, try yolo11m
# again or set YOLO_IMGSZ to 960. yolo11n is even faster but loses small
# objects badly — only use it if FPS still isn't good enough on yolo11s.
YOLO_MODEL     = os.getenv("YOLO_MODEL", "/home/anuragmishra/Anurag/Flight_detection/yolo11s.pt")
# Inference resolution. The RTSP frame is typically 1920x1080 but we don't
# need to feed YOLO at that size — 640 is the model's training size and
# slashes per-frame latency dramatically. Ultralytics handles letterboxing
# internally so detections are still in original-frame coordinates.
YOLO_IMGSZ     = int(os.getenv("YOLO_IMGSZ", "640"))

# Tracker config — BoT-SORT vs ByteTrack.
# BoT-SORT is the default because it includes camera-motion compensation,
# which matters here: the PTZ moves while it tracks, and ByteTrack assumes
# a static camera. If you ever need raw speed on a static-camera setup,
# switch to "bytetrack.yaml".
#
# IMPORTANT: pass an ABSOLUTE PATH here, not a bare filename. Ultralytics
# resolves bare filenames against its own internal cfg directory, which
# means edits to your local botsort.yaml may be silently ignored.
# Defaults to a botsort.yaml sitting next to track.py.
TRACKER_YAML   = os.getenv("TRACKER_YAML",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "botsort.yaml"))

# Coast period — how long to hold the lock when the committed ID is missing.
# 45 frames ≈ 1.5s at 30 fps. Matches the design decision for plane tracking
# (occluded by a hangar / behind a tower — wait, don't immediately re-acquire).
TRACKER_COAST_FRAMES = int(os.getenv("TRACKER_COAST_FRAMES", "45"))

# Lost-home timer — how long with no confirmed lock before we auto-return home.
# Per the design decision: 10 seconds. After this fires, the camera sends one
# home command and waits for either a new lock or another timeout.
LOST_HOME_SECONDS = float(os.getenv("LOST_HOME_SECONDS", "10.0"))
# Which home to return to on a lost-timeout:
#   "onvif"  → use the camera's ONVIF GotoHomePosition (same as 'H' key)
#   "custom" → use the home.json position (same as 'G' key)
# Per your spec: start with onvif, switch to custom later when you've
# validated the workflow. Just set LOST_HOME_TARGET=custom in your .env.
LOST_HOME_TARGET = os.getenv("LOST_HOME_TARGET", "onvif").lower()

# Manual home position — see home.json next to this script.
HOME_CONFIG_PATH = os.getenv("HOME_CONFIG_PATH",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "home.json"))

# ROI confinement — saved by the ROI mouse-drag UI.
ROI_CONFIG_PATH = os.getenv("ROI_CONFIG_PATH",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "roi.json"))

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
    
    Supports both .pt (PyTorch) and .engine (TensorRT) models:
    - .pt: uses model.track() with built-in BoT-SORT
    - .engine: uses model.predict() + manually instantiated BoT-SORT
    """
    def __init__(self, model, input_q, output_q):
        super().__init__(daemon=True)
        self.model    = model
        self.input_q  = input_q
        self.output_q = output_q
        self.running  = True
        
        # Detect model type from the file extension or model attributes
        self.is_engine = self._is_tensorrt_engine()
        
        # For .engine files, instantiate BoT-SORT manually
        self.tracker = None
        if self.is_engine:
            self.tracker = self._create_tracker()
            print(f"   ✅ Using TensorRT .engine + manual BoT-SORT tracker")
        else:
            print(f"   ✅ Using PyTorch .pt + built-in tracker")
    
    def _is_tensorrt_engine(self):
        """Check if the loaded model is a TensorRT engine."""
        # Ultralytics sets model.predictor.model_name or similar attributes
        # The simplest check: does model.track exist as a callable?
        # If not, it's likely an .engine
        try:
            # Try calling track with minimal args - if it fails immediately
            # with "not supported" error, it's an engine
            if hasattr(self.model, 'predictor'):
                predictor_name = str(type(self.model.predictor)).lower()
                if 'tensorrt' in predictor_name or 'engine' in predictor_name:
                    return True
            # Fallback: check if model path ends in .engine
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'pt_path'):
                return str(self.model.model.pt_path).endswith('.engine')
            return False
        except:
            return False
    
    def _create_tracker(self):
        """Instantiate BoT-SORT tracker manually for .engine models."""
        # Load tracker config from YAML
        with open(TRACKER_YAML) as f:
            cfg_dict = yaml.safe_load(f)
        
        # Convert to the format BoT-SORT expects
        cfg = IterableSimpleNamespace(**cfg_dict)
        
        # Create tracker instance
        tracker = BOTSORT(cfg, frame_rate=30)  # 30fps assumption
        return tracker

    def run(self):
        while self.running:
            try:
                frame = self.input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.is_engine:
                # ═══════════════════════════════════════════════════════════
                # TensorRT .engine path: model.predict() + manual BoT-SORT
                # ═══════════════════════════════════════════════════════════
                results = self.model.predict(source=frame, classes=[TRACK_CLASS],
                                            conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                                            verbose=False)
                r0 = results[0]
                
                # Convert detections to format BoT-SORT expects:
                # np.array shape (N, 6): [x1, y1, x2, y2, conf, class]
                boxes_out = []
                try:
                    if len(r0.boxes) > 0:
                        dets = []
                        for box in r0.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            dets.append([x1, y1, x2, y2, conf, cls])
                        dets = np.array(dets)
                        
                        # Run tracker update
                        tracked = self.tracker.update(dets, frame)
                        
                        # tracked is np.array shape (M, 5): [x1, y1, x2, y2, track_id]
                        # Build output matching model.track() format
                        for track in tracked:
                            x1, y1, x2, y2, tid = track
                            # Match tracked box to detection to get conf
                            # (simple heuristic: closest by center distance)
                            cx_t, cy_t = (x1 + x2) / 2, (y1 + y2) / 2
                            best_conf = YOLO_CONF  # fallback
                            min_dist = float('inf')
                            for det in dets:
                                cx_d, cy_d = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
                                dist = ((cx_t - cx_d)**2 + (cy_t - cy_d)**2)**0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    best_conf = det[4]
                            
                            boxes_out.append((
                                int(x1), int(y1), int(x2), int(y2),
                                best_conf, int(tid)
                            ))
                    else:
                        # No detections → tracker gets empty input
                        self.tracker.update(np.empty((0, 6)), frame)
                except Exception as e:
                    # If tracker crashes, log but don't kill the thread
                    print(f"⚠️  Tracker update failed: {e}")
                    # Return empty boxes for this frame
                    boxes_out = []
                
            else:
                # ═══════════════════════════════════════════════════════════
                # PyTorch .pt path: built-in model.track()
                # ═══════════════════════════════════════════════════════════
                # persist=True keeps tracker Kalman + ID state across calls.
                # Without persist, every frame starts from scratch → no stable IDs.
                #
                # The result .id field is None until tracker confirms (1-3 frames).
                # We pass None through — tracker_state.pick_track() handles it.
                results = self.model.track(frame, classes=[TRACK_CLASS],
                                          conf=YOLO_CONF, persist=True,
                                          imgsz=YOLO_IMGSZ,
                                          tracker=TRACKER_YAML, verbose=False)
                boxes_out = []
                r0 = results[0]
                ids_tensor = getattr(r0.boxes, "id", None)
                for i, box in enumerate(r0.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    if ids_tensor is not None and i < len(ids_tensor):
                        tid = int(ids_tensor[i].item())
                    else:
                        tid = None
                    boxes_out.append((int(x1), int(y1), int(x2), int(y2), conf, tid))

            # Keep only latest result
            while not self.output_q.empty():
                try: self.output_q.get_nowait()
                except: pass
            self.output_q.put(boxes_out)

    def reset_tracker(self):
        """
        Wipe the underlying tracker's state (committed IDs, Kalman states,
        appearance history). Call when toggling tracking off, going home,
        or any other moment where carrying state forward would be wrong.

        Handles both .pt (Ultralytics built-in) and .engine (manual) trackers.
        """
        try:
            if self.is_engine:
                # For manual BoT-SORT, recreate the tracker instance.
                # Safer than trying to call .reset() which may not be
                # fully implemented in the standalone BOTSORT class.
                if self.tracker is not None:
                    self.tracker = self._create_tracker()
            else:
                # For built-in .pt tracker, reach into Ultralytics internals
                predictor = getattr(self.model, "predictor", None)
                if predictor is None:
                    return
                trackers = getattr(predictor, "trackers", None)
                if not trackers:
                    return
                for t in trackers:
                    if hasattr(t, "reset"):
                        t.reset()
        except Exception as e:
            # Reset is best-effort — if Ultralytics changes internals in a
            # future release, we don't want this to kill tracking. Just warn.
            print(f"⚠️  Tracker reset failed (non-fatal): {e}")

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
            elif action == 'custom_home':
                # Move via AbsoluteMove to a user-configured position.
                # Errors here must NOT kill the thread — log and continue.
                home = cmd['home']
                try:
                    goto_custom_home(soap, self.token, home)
                except HomeConfigError as e:
                    print(f"⚠️  Custom home failed: {e}")
                except Exception as e:
                    print(f"⚠️  Custom home unexpected error: {e}")

    def stop(self):
        self.running = False

# ── Tracking helpers ──────────────────────────────────────────────────────────
def best_box(boxes):
    """Return largest box (highest area = closest bottle)."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

# ── Adaptive zoom + proportional speed config ────────────────────────────────
#
# The zoom system uses HYSTERESIS — different thresholds for "trigger zoom"
# vs "stop zoom". Without hysteresis, the box area hovers around a single
# threshold and zoom flips on/off every frame ("zoom creeping forever").
#
# Three zones:
#   box < ZOOM_IN_TRIGGER      → object is too small  → zoom in
#   ZOOM_IN_TRIGGER ≤ box ≤ ZOOM_OUT_TRIGGER → comfortable size → no zoom
#   box > ZOOM_OUT_TRIGGER     → object is too large  → zoom out
#
# The dead band is intentionally wide (1% → 25% of frame) so once the
# object is in a reasonable size range, no further zoom commands fire.

ZOOM_IN_TRIGGER  = float(os.getenv("ZOOM_IN_TRIGGER",  "0.01"))   # box <  1% of frame → too small
ZOOM_OUT_TRIGGER = float(os.getenv("ZOOM_OUT_TRIGGER", "0.25"))   # box > 25% of frame → too big

ZOOM_IN_SPD      = float(os.getenv("ZOOM_IN_SPD",  "0.10"))   # zoom-in velocity during a pulse
ZOOM_OUT_SPD     = float(os.getenv("ZOOM_OUT_SPD", "0.08"))   # zoom-out velocity during a pulse (positive — direction comes from sign in compute_zoom)
ZOOM_PULSE_MS    = int(os.getenv("ZOOM_PULSE_MS", "300"))     # pulse duration in milliseconds
ZOOM_INTERVAL    = float(os.getenv("ZOOM_INTERVAL", "1.5"))   # cooldown after pulse ends — lens settles, scene stabilizes

TRACK_SPD_MIN    = float(os.getenv("TRACK_SPD_MIN", "0.05"))   # gentler than old default (was 0.08)
TRACK_SPD_MAX    = float(os.getenv("TRACK_SPD_MAX", "0.18"))   # HALF the old 0.35 — main fix for oscillation
STEERING_CURVE   = float(os.getenv("STEERING_CURVE", "2.0"))   # 1.0 = old linear, 2.0 = quadratic (gentler), 3.0 = cubic
PAN_DIR          = int(os.getenv("PAN_DIR",  "-1"))   # flip to +1 if pan direction is wrong
TILT_DIR         = int(os.getenv("TILT_DIR", "+1"))   # flip to -1 if tilt direction is wrong

# Build the steering config once — passed into compute_steering() each frame.
# Tunable via env vars so we don't need to recompile to retune.
STEERING_CONFIG = SteeringConfig(
    dead_zone=DEAD_ZONE,           # wider dead zone reduces over-correction
    spd_min=TRACK_SPD_MIN,
    spd_max=TRACK_SPD_MAX,
    curve_exponent=STEERING_CURVE,  # quadratic by default
    pan_dir=PAN_DIR,
    tilt_dir=TILT_DIR,
)

def offset_to_ptz(cx_obj, cy_obj, w, h):
    """
    Thin wrapper for backwards-compat. Delegates to compute_steering()
    in steering.py — see that module for the actual control law.

    The old linear-ramp body was removed because it caused observable
    oscillation in field tests (see commit notes). The new quadratic
    curve in steering.py is the single biggest fix in this revision.
    """
    return compute_steering(cx_obj, cy_obj, w, h, STEERING_CONFIG)

def compute_zoom(box_area_ratio, last_zoom_time, currently_pulsing):
    """
    Decide whether to start a new zoom pulse this frame.

    Pulse model
    -----------
    A "pulse" is a short, time-bounded zoom command:
      1. issue zoom at ±ZOOM_IN_SPD for ZOOM_PULSE_MS milliseconds
      2. issue zoom=0 explicitly (this is what was missing — previously
         the camera would zoom continuously because no stop was ever sent)
      3. wait ZOOM_INTERVAL seconds for the lens to physically settle
      4. re-evaluate

    Hysteresis prevents flapping: we use two distinct thresholds so the
    box must move significantly to flip from "zoom in" to "no zoom" or
    vice versa.

    Parameters
    ----------
    box_area_ratio   : current box area / frame area
    last_zoom_time   : timestamp when the last pulse FINISHED (or 0)
    currently_pulsing: True if we're inside an active pulse and shouldn't
                       start another one

    Returns
    -------
    (zoom_velocity, should_send_new_pulse)
      zoom_velocity        — value to send to PTZ ContinuousMove for this pulse
      should_send_new_pulse — True only when a new pulse should be issued NOW
    """
    if currently_pulsing:
        return 0.0, False

    now = time.time()
    if (now - last_zoom_time) < ZOOM_INTERVAL:
        return 0.0, False     # still in cooldown after last pulse

    if box_area_ratio < ZOOM_IN_TRIGGER:
        # Object too small — zoom in
        return float(ZOOM_IN_SPD), True

    if box_area_ratio > ZOOM_OUT_TRIGGER:
        # Object too big — zoom out (negative velocity)
        return -float(ZOOM_OUT_SPD), True

    # Inside the comfortable zone — no zoom needed
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
    ctrl = ("T:Manual  SPACE:Stop  H/G:Home  P:Pos  R:SaveHome  I:ROI  C:ClearROI  Q:Quit"
            if mode == "TRACK"
            else "W/S/A/D:Move  Z/X:Zoom  T:Track  H/G:Home  P:Pos  R:SaveHome  I:ROI  C:ClearROI  Q:Quit")
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

    # Tracker YAML — verify we're loading the file we THINK we're loading.
    # If TRACKER_YAML points to a non-existent file, Ultralytics silently
    # falls back to its bundled default — which is the exact symptom you
    # were seeing. Fail loudly here instead.
    print(f"🎯 Tracker config: {TRACKER_YAML}")
    if not os.path.isfile(TRACKER_YAML):
        raise FileNotFoundError(
            f"Tracker YAML not found at: {TRACKER_YAML}\n"
            f"   Either place botsort.yaml next to track.py, or set "
            f"TRACKER_YAML to an absolute path."
        )
    print("   ✅ Tracker YAML found\n")

    # Camera
    print("🔌 Connecting to camera...")
    token = get_token()
    print(f"✅ Profile: {token}")
    _set_home(token)

    # Manual home position — config file the user edits with runway coordinates
    print(f"📍 Loading home config: {HOME_CONFIG_PATH}")
    try:
        home_pos = HomePosition.load_or_default(HOME_CONFIG_PATH)
        if home_pos.is_default:
            print("   ⚠️  home.json not found — custom home defaults to (0,0,0).")
            print("      Aim camera at runway, press 'P' to read coords, then write home.json.")
        else:
            print(f"   ✅ Custom home: pan={home_pos.pan} tilt={home_pos.tilt} zoom={home_pos.zoom}")
    except HomeConfigError as e:
        print(f"   ❌ Home config error: {e}")
        print("      Falling back to (0,0,0). Fix home.json or unset HOME_CONFIG_PATH.")
        home_pos = HomePosition(0.0, 0.0, 0.0, is_default=True)

    # ROI — optional region of interest for detection filtering
    print(f"🔲 Loading ROI config: {ROI_CONFIG_PATH}")
    try:
        roi = ROI.load_or_none(ROI_CONFIG_PATH)
        if roi is None:
            print("   ℹ️  No ROI set — tracking will consider the entire frame.")
            print("      Press 'I' then drag mouse on the live feed to set an ROI.")
        else:
            print(f"   ✅ ROI: ({roi.x1:.2f},{roi.y1:.2f}) → ({roi.x2:.2f},{roi.y2:.2f}) "
                  f"[{roi.area_fraction()*100:.1f}% of frame]")
    except ROIError as e:
        print(f"   ❌ ROI config error: {e}")
        print("      Continuing without ROI. Fix roi.json or delete it.")
        roi = None

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

    # ── ROI drag state (mutable via mouse callback) ──────────────────────
    # We use a dict so the closure inside on_mouse can mutate fields without
    # the nonlocal-in-callback dance. Mouse coords are in the *current frame's*
    # pixel space — convertedfraction only at save time.
    roi_drag = {
        "active":  False,    # ROI selection mode (toggled by 'I')
        "start":   None,     # (x, y) where the user began dragging
        "current": None,     # (x, y) where the mouse currently is during drag
        "dragging": False,   # left mouse button is down
        "frame_size": (0, 0),  # (w, h) of the most recent frame — used at save time
    }

    def on_mouse(event, x, y, flags, _):
        # The mouse callback fires whether we're in ROI mode or not.
        # Only react when we have explicitly enabled ROI selection.
        if not roi_drag["active"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_drag["start"]    = (x, y)
            roi_drag["current"]  = (x, y)
            roi_drag["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and roi_drag["dragging"]:
            roi_drag["current"]  = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and roi_drag["dragging"]:
            roi_drag["dragging"] = False
            # Save handled in main loop (need access to `roi`/`ROI_CONFIG_PATH`)
            # — we leave the start/current points in place as a signal.

    cv2.setMouseCallback(WIN, on_mouse)

    # State
    track_mode     = False
    current_action = None
    status         = "READY"
    last_box_data  = None    # (x1,y1,x2,y2,conf) — only updated when fresh detection
    stale_counter  = 0
    last_track_cmd = (0.0, 0.0, 0.0)
    last_zoom_time = 0.0    # timestamp when the last zoom pulse FINISHED (cooldown anchor)
    zoom_pulse_until = 0.0  # timestamp when the current zoom pulse should END (0 = no active pulse)
    frame_idx      = 0
    fps_t          = time.time()
    fps            = 0.0

    # ID-aware tracker state. Survives across the main loop; reset on
    # T-toggle, H-home, G-custom-home, and tracker disable to avoid
    # carrying stale IDs into a new tracking session.
    tracker_state  = TrackerState(coast_frames=TRACKER_COAST_FRAMES)

    # Lost-target auto-home timer. Ticks while there's no committed
    # tracker ID. Fires a single home command after LOST_HOME_SECONDS.
    lost_home_timer = LostHomeTimer(threshold_seconds=LOST_HOME_SECONDS)

    print("Phase 2 — Live feed active. Click the window, then press keys.\n")

    while True:
        frame = cap_thread.get_frame()
        if frame is None:
            continue

        # Flip (camera upside-down)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w  = frame.shape[:2]
        frame_idx += 1
        roi_drag["frame_size"] = (w, h)

        # ── Process pending ROI mouse-drag completion ────────────────────
        # If the user released the mouse, finalize the ROI selection.
        # We do this in the main loop (not the callback) because it touches
        # `roi` and the file system — clearer than locking from the callback.
        if (roi_drag["active"] and not roi_drag["dragging"]
                and roi_drag["start"] is not None and roi_drag["current"] is not None):
            sx, sy = roi_drag["start"]
            ex, ey = roi_drag["current"]
            # A single click without movement → still treat as "abort"
            if abs(ex - sx) > 5 and abs(ey - sy) > 5:
                try:
                    new_roi = ROI.from_pixels(sx, sy, ex, ey, w, h)
                    new_roi.save(ROI_CONFIG_PATH)
                    roi = new_roi
                    print(f"🔲 ROI set: ({new_roi.x1:.2f},{new_roi.y1:.2f}) → "
                          f"({new_roi.x2:.2f},{new_roi.y2:.2f}) — saved to {ROI_CONFIG_PATH}")
                    status = "ROI SET"
                except ROIError as e:
                    print(f"⚠️  ROI rejected: {e}")
                    status = "ROI TOO SMALL"
            # Either way, exit ROI mode after a release
            roi_drag["active"]  = False
            roi_drag["start"]   = None
            roi_drag["current"] = None

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
                tracks = yolo_out_q.get_nowait()
                # ── ROI filter: drop detections outside the user-defined ROI ──
                # Track tuples are (x1,y1,x2,y2,conf,id). filter_boxes_in_roi
                # only looks at the first four fields, so it works unchanged
                # on the extended tuple. No-op when roi is None.
                tracks = filter_boxes_in_roi(tracks, roi, w, h)
                # pick_track replaces the old best_box(). It knows about
                # track IDs, identity stability, and the 45-frame coast.
                chosen = pick_track(tracks, tracker_state, w, h)
                if chosen is not None:
                    # Drop the track_id for downstream code — last_box_data
                    # is still a 5-tuple (x1,y1,x2,y2,conf) so the steering
                    # logic below doesn't change.
                    last_box_data = chosen[:5]
                    stale_counter = 0
                else:
                    # CRITICAL: clear last_box_data the moment pick_track returns
                    # None. Before this fix, the camera would keep steering toward
                    # the previous frame's box for STALE_FRAMES (15) more frames
                    # — even though the tracker was already telling us "no lock."
                    # That's the "moves randomly and stops somewhere" symptom:
                    # camera chases the location where the bottle USED to be.
                    last_box_data = None
                    stale_counter += 1
            except queue.Empty:
                pass

        # ── Tracking PTZ control ──────────────────────────────────────────
        if track_mode:
            # Handle end-of-pulse FIRST — runs every frame, independent of
            # whether we have a target. This is what stops the camera from
            # zooming forever: when a pulse expires we send pan/tilt-only
            # (zoom=0), or full stop if neither pan nor tilt is active.
            if zoom_pulse_until > 0 and time.time() >= zoom_pulse_until:
                zoom_pulse_until = 0
                last_zoom_time = time.time()    # start cooldown from pulse end
                # Re-send the current pan/tilt but with zoom=0 to cancel
                # the continuous zoom command. If we have no target, full stop.
                if last_box_data is not None:
                    cur_pan, cur_tilt, _ = last_track_cmd
                    if cur_pan == 0.0 and cur_tilt == 0.0:
                        ptz_thread.send({'action': 'stop'})
                    else:
                        ptz_thread.send({'action': 'move',
                                         'pan': cur_pan, 'tilt': cur_tilt,
                                         'zoom': 0.0})
                    last_track_cmd = (cur_pan, cur_tilt, 0.0)
                else:
                    ptz_thread.send({'action': 'stop'})
                    last_track_cmd = (0.0, 0.0, 0.0)

            if last_box_data:
                x1, y1, x2, y2, conf = last_box_data
                cx_obj = (x1 + x2) // 2
                cy_obj = (y1 + y2) // 2

                # Draw box + line to crosshair
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"PERSON {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.line(frame, (w//2, h//2), (cx_obj, cy_obj), (0, 255, 255), 1)

                pan, tilt = offset_to_ptz(cx_obj, cy_obj, w, h)

                # Compute whether to START a new zoom pulse this frame.
                # compute_zoom is hysteretic and respects the cooldown.
                box_area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                currently_pulsing = zoom_pulse_until > 0
                zoom, start_new_pulse = compute_zoom(box_area_ratio,
                                                    last_zoom_time,
                                                    currently_pulsing)

                # Pick the zoom value to actually send this frame:
                #   - if we just decided to start a pulse: use the new pulse zoom
                #   - if we're inside an active pulse: keep the existing zoom value
                #   - otherwise: zoom=0
                if start_new_pulse:
                    effective_zoom = zoom
                    zoom_pulse_until = time.time() + (ZOOM_PULSE_MS / 1000.0)
                elif currently_pulsing:
                    effective_zoom = last_track_cmd[2]
                else:
                    effective_zoom = 0.0

                # Draw size indicator next to box
                pct = box_area_ratio * 100
                if effective_zoom > 0:
                    z_txt = f"ZOOM IN  {pct:.1f}%"
                    z_col = (0, 255, 255)
                elif effective_zoom < 0:
                    z_txt = f"ZOOM OUT {pct:.1f}%"
                    z_col = (0, 165, 255)
                else:
                    z_txt = f"OK {pct:.1f}%"
                    z_col = (100, 255, 100)
                cv2.putText(frame, z_txt, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, z_col, 1)

                # Decide whether to (re-)send the move command this frame.
                # We send when:
                #   - pan/tilt direction changed (including dropping to 0,0
                #     when entering the dead zone — this STOPS the camera), OR
                #   - we just started a new zoom pulse
                # Sending every frame would flood the SOAP queue.
                pt_changed = (pan, tilt) != last_track_cmd[:2]

                # Always update status every frame (independent of whether
                # we send a command). The old code only set status inside
                # the "if pt_changed" block, so the status text went stale
                # between command changes — producing the "TRACK ??? ???"
                # symptom seen in field testing.
                if pan == 0.0 and tilt == 0.0 and effective_zoom == 0.0:
                    status = "CENTERED ✓"
                else:
                    dirs = []
                    if tilt > 0: dirs.append("↓")
                    if tilt < 0: dirs.append("↑")
                    if pan > 0: dirs.append("→" if PAN_DIR > 0 else "←")
                    if pan < 0: dirs.append("←" if PAN_DIR > 0 else "→")
                    if effective_zoom > 0: dirs.append("Z+")
                    if effective_zoom < 0: dirs.append("Z-")
                    status = "TRACK " + " ".join(dirs) if dirs else "TRACK"

                if pt_changed or start_new_pulse:
                    if pan == 0.0 and tilt == 0.0 and effective_zoom == 0.0:
                        # Entering the dead zone — explicit stop so the
                        # camera doesn't coast on the previous velocity.
                        ptz_thread.send({'action': 'stop'})
                        last_track_cmd = (0.0, 0.0, 0.0)
                    else:
                        last_track_cmd = (pan, tilt, effective_zoom)
                        ptz_thread.send({'action': 'move', 'pan': pan,
                                         'tilt': tilt, 'zoom': effective_zoom})

            else:
                # No target — stop if we were moving (and cancel any active pulse)
                if last_track_cmd != (0.0, 0.0, 0.0):
                    last_track_cmd = (0.0, 0.0, 0.0)
                    zoom_pulse_until = 0
                    ptz_thread.send({'action': 'stop'})
                status = "SEARCHING..."

            # ── Lost-target auto-home ────────────────────────────────────
            # Wall-clock timer: if tracker_state has no committed ID for
            # LOST_HOME_SECONDS seconds, send one home command. This is the
            # "give up and reset" behavior — for airport use, the camera
            # returns to the runway threshold and waits for the next plane.
            #
            # "locked" = committed_id is not None — same definition the rest
            # of the tracker uses. This composes with the coast period
            # cleanly: while coasting, we ARE locked (committed_id holds),
            # so the lost-timer doesn't tick.
            lost_home_timer.update(
                locked=(tracker_state.committed_id is not None)
            )
            if lost_home_timer.consume_home_signal():
                print(f"⏰ No target for {LOST_HOME_SECONDS}s — returning home "
                      f"(target={LOST_HOME_TARGET}).")
                if LOST_HOME_TARGET == "custom" and not home_pos.is_default:
                    ptz_thread.send({'action': 'custom_home', 'home': home_pos})
                    status = "AUTO HOME (custom)"
                else:
                    ptz_thread.send({'action': 'home'})
                    status = "AUTO HOME (onvif)"
                # Reset everything so we start tracking fresh from home
                tracker_state.reset()
                yolo_thread.reset_tracker()
                last_box_data = None
                last_track_cmd = (0.0, 0.0, 0.0)
                zoom_pulse_until = 0

        # ── ROI overlay ──────────────────────────────────────────────────
        # Draw the saved ROI (if any) as a green rectangle so the user
        # always knows the active region. Detections outside it are filtered
        # upstream, so we don't need a "dimmed" alternative.
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi.to_pixels(w, h)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 200, 0), 2)
            cv2.putText(frame, "ROI", (rx1 + 4, ry1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        # In-progress drag rectangle (drawn in cyan)
        if roi_drag["active"] and roi_drag["start"] and roi_drag["current"]:
            sx, sy = roi_drag["start"]
            ex, ey = roi_drag["current"]
            cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 200, 0), 2)
            cv2.putText(frame, "Drag to set ROI — release to save",
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        elif roi_drag["active"]:
            cv2.putText(frame, "ROI MODE — drag mouse to select region",
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

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
            zoom_pulse_until = 0.0      # cancel any in-flight zoom pulse
            current_action = None
            # Clear YOLO queues
            while not yolo_in_q.empty():
                try: yolo_in_q.get_nowait()
                except: pass
            while not yolo_out_q.empty():
                try: yolo_out_q.get_nowait()
                except: pass
            # Reset both layers of tracker state:
            #  - tracker_state: our committed-ID + miss-counter
            #  - yolo_thread:   BoT-SORT's internal Kalman / ID history
            #  - lost_home_timer: don't auto-home immediately after a toggle
            # Doing one without the other would leak state across sessions.
            tracker_state.reset()
            yolo_thread.reset_tracker()
            lost_home_timer.reset()
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
            zoom_pulse_until = 0.0
            current_action = 'home'
            status         = "HOME"
            # Going home changes the scene — old track IDs are meaningless.
            tracker_state.reset()
            yolo_thread.reset_tracker()
            lost_home_timer.reset()
            ptz_thread.send({'action': 'home'})

        elif key == ord('g'):
            # Custom home — move to user-configured runway position
            track_mode     = False
            last_box_data  = None
            last_track_cmd = (0.0, 0.0, 0.0)
            zoom_pulse_until = 0.0
            current_action = 'custom_home'
            # Same reasoning as 'H' — scene change invalidates IDs.
            tracker_state.reset()
            yolo_thread.reset_tracker()
            lost_home_timer.reset()
            if home_pos.is_default:
                status = "CUSTOM HOME (default 0,0,0)"
                print("🏁 Going to custom home (default — no home.json found)")
            else:
                status = f"CUSTOM HOME ({home_pos.pan:+.2f},{home_pos.tilt:+.2f},{home_pos.zoom:+.2f})"
                print(f"🏁 Going to custom home: pan={home_pos.pan} tilt={home_pos.tilt} zoom={home_pos.zoom}")
            ptz_thread.send({'action': 'custom_home', 'home': home_pos})

        elif key == ord('p'):
            # Print current PTZ position so user can paste into home.json.
            # Done synchronously (blocks ~50ms) — acceptable since user
            # presses this only while aiming, not during live tracking.
            try:
                pan, tilt, zoom = get_current_position(soap, token)
                print(f"\n📍 Current PTZ position:")
                print(f'   {{"pan": {pan:.4f}, "tilt": {tilt:.4f}, "zoom": {zoom:.4f}}}')
                print(f"   ↑ paste this into {os.path.basename(HOME_CONFIG_PATH)}\n")
                status = f"POS ({pan:+.2f},{tilt:+.2f},{zoom:+.2f})"
            except HomeConfigError as e:
                print(f"⚠️  Could not read position: {e}")
                status = "POS READ FAILED"
            except Exception as e:
                print(f"⚠️  Unexpected error reading position: {e}")
                status = "POS READ FAILED"

        elif key == ord('r'):
            # Capture current PTZ → save as new custom home (backup the old one).
            try:
                pan, tilt, zoom = get_current_position(soap, token)
                new_home = HomePosition(pan, tilt, zoom)
                backup_path = backup_then_save(new_home, HOME_CONFIG_PATH)
                # Refresh in-memory state so 'G' uses the new home immediately
                home_pos = new_home
                if backup_path:
                    print(f"\n💾 Saved new custom home: pan={pan:.4f} tilt={tilt:.4f} zoom={zoom:.4f}")
                    print(f"   Previous home backed up to {backup_path}\n")
                else:
                    print(f"\n💾 Saved new custom home: pan={pan:.4f} tilt={tilt:.4f} zoom={zoom:.4f}")
                    print(f"   (no previous home.json, so no .bak created)\n")
                status = f"HOME SAVED ({pan:+.2f},{tilt:+.2f},{zoom:+.2f})"
            except HomeConfigError as e:
                print(f"⚠️  Could not save home: {e}")
                status = "HOME SAVE FAILED"
            except Exception as e:
                print(f"⚠️  Unexpected error saving home: {e}")
                status = "HOME SAVE FAILED"

        elif key == ord('i'):
            # Toggle ROI selection mode. Disables tracking while active to
            # avoid PTZ jitter during a click-drag.
            if roi_drag["active"]:
                # Pressing 'I' again cancels selection
                roi_drag["active"]   = False
                roi_drag["start"]    = None
                roi_drag["current"]  = None
                roi_drag["dragging"] = False
                status = "ROI MODE OFF"
                print("🔲 ROI selection cancelled")
            else:
                track_mode = False
                ptz_thread.send({'action': 'stop'})
                roi_drag["active"] = True
                status = "ROI MODE — drag on frame"
                print("🔲 ROI selection mode — drag the mouse on the video to select a region")

        elif key == ord('c'):
            # Clear current ROI (and delete roi.json so it stays cleared after restart)
            if roi is None:
                status = "NO ROI TO CLEAR"
                print("ℹ️  No ROI is currently set.")
            else:
                roi = None
                try:
                    if os.path.isfile(ROI_CONFIG_PATH):
                        os.remove(ROI_CONFIG_PATH)
                        print(f"🗑️  ROI cleared (deleted {ROI_CONFIG_PATH})")
                    else:
                        print("🗑️  ROI cleared")
                except OSError as e:
                    print(f"⚠️  Could not delete {ROI_CONFIG_PATH}: {e}")
                status = "ROI CLEARED"

        elif key == ord(' '):
            track_mode     = False
            last_box_data  = None
            last_track_cmd = (0.0, 0.0, 0.0)
            zoom_pulse_until = 0.0
            current_action = 'stop'
            status         = "STOPPED"
            # Stop means stop — clear track state too so when user re-enables
            # tracking it's a fresh session.
            tracker_state.reset()
            yolo_thread.reset_tracker()
            lost_home_timer.reset()
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