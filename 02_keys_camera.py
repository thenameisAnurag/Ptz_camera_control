"""
PTZ Camera Keyboard Control + Live RTSP Feed
=============================================
Controls (click the video window first to focus it):
  W / S     → Tilt UP / DOWN
  A / D     → Pan LEFT / RIGHT
  Z / X     → Zoom IN / OUT
  SPACE     → STOP movement
  H         → Go to HOME (origin)
  Q         → QUIT

Run: python 02_keys_camera.py
Requires: pip install -r requirements.txt
"""

import os
import requests
from requests.auth import HTTPDigestAuth
import urllib3
import xml.etree.ElementTree as ET
import time
import sys
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings()

# ── Config ───────────────────────────────────────────────────────────────────
IP       = os.getenv("CAMERA_IP")
USERNAME = os.getenv("CAMERA_USERNAME")
PASSWORD = os.getenv("CAMERA_PASSWORD")

if not all([IP, USERNAME, PASSWORD]):
    raise RuntimeError("Missing camera credentials. Copy .env.example → .env and fill in values.")

URL      = f"https://{IP}/onvif/device_service"
AUTH     = HTTPDigestAuth(USERNAME, PASSWORD)
HEADERS  = {"Content-Type": "application/soap+xml"}
SPEED    = float(os.getenv("PTZ_SPEED", "0.4"))
ZOOM_SPD = float(os.getenv("ZOOM_SPEED", "0.3"))

# ── SOAP helper ───────────────────────────────────────────────────────────────
def soap(body: str) -> str:
    envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
  <s:Body>{body}</s:Body>
</s:Envelope>"""
    r = requests.post(URL, data=envelope, headers=HEADERS,
                      auth=AUTH, verify=False, timeout=10)
    return r.text

# ── ONVIF calls ───────────────────────────────────────────────────────────────
def get_token() -> str:
    resp = soap('<GetProfiles xmlns="http://www.onvif.org/ver10/media/wsdl"/>')
    root = ET.fromstring(resp)
    profiles = root.findall('.//{http://www.onvif.org/ver10/media/wsdl}Profiles')
    return profiles[0].attrib['token']

def get_rtsp_url(token: str) -> str:
    body = f"""<GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl">
  <StreamSetup>
    <Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>
    <Transport xmlns="http://www.onvif.org/ver10/schema">
      <Protocol>RTSP</Protocol>
    </Transport>
  </StreamSetup>
  <ProfileToken>{token}</ProfileToken>
</GetStreamUri>"""
    resp = soap(body)
    root = ET.fromstring(resp)
    uri  = root.find('.//{http://www.onvif.org/ver10/schema}Uri')
    if uri is not None:
        raw = uri.text
        # Inject credentials into the RTSP URL
        return raw.replace("rtsp://", f"rtsp://{USERNAME}:{PASSWORD}@")
    return None

def ptz_move(token: str, pan=0.0, tilt=0.0, zoom=0.0):
    body = f"""<ContinuousMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
  <Velocity>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema" x="{pan}" y="{tilt}"/>
    <Zoom    xmlns="http://www.onvif.org/ver10/schema" x="{zoom}"/>
  </Velocity>
</ContinuousMove>"""
    soap(body)

def ptz_stop(token: str):
    body = f"""<Stop xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
  <PanTilt>true</PanTilt>
  <Zoom>true</Zoom>
</Stop>"""
    soap(body)

def go_home(token: str):
    body = f"""<GotoHomePosition xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
  <Speed>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema" x="0.5" y="0.5"/>
    <Zoom    xmlns="http://www.onvif.org/ver10/schema" x="0.5"/>
  </Speed>
</GotoHomePosition>"""
    resp = soap(body)
    if "Fault" in resp:
        print("⚠️  No home set — stopping instead.")
        ptz_stop(token)

def set_home(token: str):
    body = f"""<SetHomePosition xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
</SetHomePosition>"""
    soap(body)

# ── Phase 1: Auto movement test ───────────────────────────────────────────────
def run_auto_test(token: str):
    print("\n" + "="*50)
    print("  PHASE 1 — Auto Movement Test (5s each direction)")
    print("="*50)
    steps = [
        ("⬅️  LEFT",  dict(pan=-SPEED)),
        ("➡️  RIGHT", dict(pan= SPEED)),
        ("⬆️  UP",    dict(tilt= SPEED)),
        ("⬇️  DOWN",  dict(tilt=-SPEED)),
    ]
    for label, kwargs in steps:
        print(f"\n{label} for 5 seconds...")
        ptz_move(token, **kwargs)
        time.sleep(5)
        ptz_stop(token)
        print("   🛑 Stopped")
        time.sleep(1)

    print("\n🏠 Returning to origin...")
    go_home(token)
    time.sleep(4)
    print("✅ Auto test complete!\n")

# ── HUD overlay ───────────────────────────────────────────────────────────────
def draw_hud(frame, status: str):
    h, w = frame.shape[:2]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    # Bottom bar
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar2, 0.55, frame, 0.45, 0, frame)

    # Status
    color = (0, 255, 100) if status == "READY" or status == "STOPPED" else (0, 200, 255)
    cv2.putText(frame, f"PTZ  |  {status}", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # LIVE dot
    cv2.circle(frame, (w - 22, 20), 8, (0, 0, 220), -1)
    cv2.putText(frame, "LIVE", (w - 60, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

    # Controls bottom
    ctrl = "W/S: Tilt    A/D: Pan    Z/X: Zoom    SPACE: Stop    H: Home    Q: Quit"
    cv2.putText(frame, ctrl, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame

# ── Key map ───────────────────────────────────────────────────────────────────
KEY_MAP = {
    ord('w'): ("TILT UP",    dict(tilt= SPEED)),
    ord('s'): ("TILT DOWN",  dict(tilt=-SPEED)),
    ord('a'): ("PAN LEFT",   dict(pan =-SPEED)),
    ord('d'): ("PAN RIGHT",  dict(pan = SPEED)),
    ord('z'): ("ZOOM IN",    dict(zoom= ZOOM_SPD)),
    ord('x'): ("ZOOM OUT",   dict(zoom=-ZOOM_SPD)),
}

# ── Phase 2: Live feed + keyboard ────────────────────────────────────────────
def run_with_feed(token: str, rtsp_url: str):
    print(f"\n📹 Opening RTSP stream...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("⚠️  Retrying without embedded credentials...")
        fallback = rtsp_url.replace(f"{USERNAME}:{PASSWORD}@", "")
        cap = cv2.VideoCapture(fallback)

    if not cap.isOpened():
        print("❌ Could not open RTSP stream.")
        return False

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # low latency
    print("✅ Stream open! Click the video window then use keys.\n")

    WIN = "PTZ Camera Control  |  Click here then press keys"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    current_action = None
    status         = "READY"

    while True:
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Stream lost — reconnecting...",
                        (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            status = "NO SIGNAL"
        else:
            draw_hud(frame, status)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ──────────────────────────────────────────────────────
        if key == ord('q'):
            print("\n👋 Quitting...")
            ptz_stop(token)
            break

        elif key == ord('h'):
            if current_action != 'home':
                current_action = 'home'
                status         = "HOME"
                print("🏠 Going home...")
                go_home(token)

        elif key == ord(' '):
            if current_action != 'stop':
                current_action = 'stop'
                status         = "STOPPED"
                print("🛑 STOP")
                ptz_stop(token)

        elif key in KEY_MAP:
            label, kwargs = KEY_MAP[key]
            if current_action != key:
                current_action = key
                status         = label
                print(f"  ▶ {label}")
                ptz_move(token, **kwargs)

        elif key == 0xFF:
            # No key pressed — release movement
            if current_action not in (None, 'stop', 'home'):
                current_action = None
                status         = "STOPPED"
                ptz_stop(token)
                print("  🛑 Released")

    cap.release()
    cv2.destroyAllWindows()
    return True

# ── Fallback: keyboard lib only (no video) ────────────────────────────────────
def run_keyboard_only(token: str):
    try:
        import keyboard as kb
    except ImportError:
        print("❌ pip install keyboard  (run with sudo on Linux)")
        return

    print("⌨️  Keyboard-only mode. W/S/A/D/Z/X / SPACE / H / Q\n")
    STR_MAP = {
        'w': dict(tilt= SPEED),  's': dict(tilt=-SPEED),
        'a': dict(pan =-SPEED),  'd': dict(pan = SPEED),
        'z': dict(zoom= ZOOM_SPD), 'x': dict(zoom=-ZOOM_SPD),
    }
    current = None
    try:
        while True:
            pressed = next((k for k in STR_MAP if kb.is_pressed(k)), None)
            if kb.is_pressed('q'):
                ptz_stop(token); break
            elif kb.is_pressed('h'):
                if current != 'home': current = 'home'; go_home(token)
            elif kb.is_pressed('space'):
                if current != 'stop': current = 'stop'; ptz_stop(token)
            elif pressed:
                if current != pressed:
                    current = pressed; ptz_move(token, **STR_MAP[pressed])
            else:
                if current not in (None, 'stop', 'home'):
                    current = None; ptz_stop(token)
            time.sleep(0.1)
    except KeyboardInterrupt:
        ptz_stop(token)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔌 Connecting to camera...")
    token = get_token()
    print(f"✅ Connected — profile: {token}")

    print("💾 Saving current position as HOME...")
    set_home(token)

    rtsp_url = get_rtsp_url(token)
    print(f"📹 RTSP URL: {rtsp_url}" if rtsp_url else "⚠️  No RTSP URL found")

    # Phase 1 — auto movement test
    run_auto_test(token)

    # Phase 2 — live feed + keyboard control
    print("\n" + "="*50)
    print("  PHASE 2 — Live Feed + Keyboard Control")
    print("="*50)

    if rtsp_url:
        ok = run_with_feed(token, rtsp_url)
        if not ok:
            run_keyboard_only(token)
    else:
        run_keyboard_only(token)
