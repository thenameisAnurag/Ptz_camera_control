# 🎯 PTZ Camera Tracker — *Eyes that follow*

> A threaded, GPU-accelerated ONVIF PTZ controller that locks onto objects in real-time using YOLOv8.
> Move it with your keyboard. Or let the AI take the wheel.

<p align="center">
  <img alt="status"    src="https://img.shields.io/badge/status-active-brightgreen">
  <img alt="python"    src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="cuda"      src="https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia">
  <img alt="onvif"     src="https://img.shields.io/badge/protocol-ONVIF-orange">
  <img alt="opencv"    src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv">
  <img alt="yolo"      src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B00">
  <img alt="docker"    src="https://img.shields.io/badge/docker-ready-2496ED?logo=docker">
  <img alt="license"   src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

## ✨ What is this?

Four progressive scripts that turn an off-the-shelf ONVIF PTZ camera into an **intelligent visual tracker**:

| File | What it does | When to run it |
|---|---|---|
| 🔌 `01_camera.py`         | Sanity test — connects, runs a scripted L/R/U/D/zoom routine | First-time setup, "is the camera alive?" |
| 🎮 `02_keys_camera.py`    | Live RTSP feed + WASD keyboard control + on-screen HUD       | When you just want to drive it manually |
| 🤖 `03_tracking.py`       | 3-thread architecture: capture · GPU YOLO · PTZ · display    | Auto-track a single class (default: bottle) |
| 🚀 `04_updated_track.py`  | Same, plus **adaptive zoom**, **proportional speed**, **motion dead-zone** | The "good one" — production tracker |

---

## 🏗️ Architecture

```
                 ┌────────────────┐
   RTSP H.264 ──▶│ CaptureThread  │──▶ latest_frame  ┐
                 └────────────────┘                  │
                                                     ▼
   ┌──────────────┐    frame    ┌────────────┐   draw + decide
   │  YOLOThread  │◀────────────│ Main loop  │──▶ ┌─────────────┐
   │  (GPU)       │──▶ boxes ──▶│ (cv2 + UI) │   │ HUD overlay │
   └──────────────┘             └─────┬──────┘   └─────────────┘
                                      │
                          PTZ command │  (drop-old queue)
                                      ▼
                              ┌──────────────┐    SOAP/HTTPS
                              │  PTZThread   │──────────────▶ 📷
                              └──────────────┘
```

Why threaded? Three rules:

1. **Display never waits for the camera** — buffer-of-1 capture eliminates RTSP lag.
2. **Display never waits for YOLO** — inference runs in parallel; main thread reads the latest result.
3. **Display never waits for the network** — PTZ SOAP calls go to a queue that drops stale commands.

Result: smooth ~30 FPS preview, low-latency tracking, no GUI freezes.

---

## 🚀 Quickstart

### 1. Clone & configure

```bash
git clone https://github.com/<you>/ptz-camera-tracking.git
cd ptz-camera-tracking
cp .env.example .env
# then edit .env with your camera's IP, username, password
```

### 2. Install (local)

```bash
# CUDA wheels (recommended — much faster YOLO inference)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install -r requirements.txt
```

### 3. Run

```bash
python 01_camera.py             # smoke test
python 02_keys_camera.py        # keyboard + live feed
python 03_tracking.py           # basic auto-track
python 04_updated_track.py      # adaptive-zoom auto-track ★
```

> 💡 Click the **video window** before pressing keys — OpenCV reads keyboard input only when its window has focus.

---

## 🎮 Controls

| Key         | Action                                  |
|:-----------:|------------------------------------------|
| `W` / `S`   | Tilt up / down                           |
| `A` / `D`   | Pan left / right                         |
| `Z` / `X`   | Zoom in / out                            |
| `T`         | **Toggle auto-tracking** (scripts 3 & 4) |
| `SPACE`     | Stop all motion                          |
| `H`         | Go to home position                      |
| `Q`         | Quit                                     |

---

## 🧠 How tracking works (the short version)

For each frame:

1. **Detect** — YOLOv8 runs on the GPU and returns bounding boxes for the target class.
2. **Pick** — the largest box (proxy for "closest / most relevant target") wins.
3. **Steer** — convert the box's offset from the frame center into normalized `(pan, tilt)` velocities.
4. **Dead-zone** — if the target is already near center, do nothing (prevents jitter).
5. **Adaptive zoom** *(script 04)* — if the box is too small in the frame → zoom in; too large → zoom out. Cooldown prevents seesawing.

The sign convention is verified for one specific camera; flip `PAN_DIR` / `TILT_DIR` in the script if your camera tracks the wrong way.

---

## 🐳 Docker

Build:

```bash
docker build -t ptz-tracker .
```

Run with GPU + X11 forwarding (so the OpenCV window appears on your host):

```bash
xhost +local:docker

docker run --rm -it --gpus all \
  --env-file .env \
  -v $(pwd)/yolov8n.pt:/app/yolov8n.pt \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network host \
  ptz-tracker python 04_updated_track.py
```

Headless / inference-only? Strip the X11 mounts and run script `01` or pipe frames to a file.

---

## ⚙️ Configuration

All knobs live in `.env` (see [`.env.example`](.env.example)):

| Variable        | Default         | Meaning                                                    |
|---------------- |---------------- |------------------------------------------------------------|
| `CAMERA_IP`     | —               | Camera IP address                                          |
| `CAMERA_USERNAME` | —             | ONVIF user                                                 |
| `CAMERA_PASSWORD` | —             | ONVIF password                                             |
| `PTZ_SPEED`     | `0.35`          | Manual pan/tilt speed                                      |
| `ZOOM_SPEED`    | `0.3`           | Manual zoom speed                                          |
| `TRACK_SPEED`   | `0.25`          | Auto-track baseline speed                                  |
| `DEAD_ZONE`     | `0.10`          | Fraction of frame around center where tracker stays still  |
| `TRACK_CLASS`   | `39` (bottle)   | COCO class id to follow (`0`=person, `4`=airplane, etc.)   |
| `YOLO_CONF`     | `0.20`          | Detection confidence threshold                             |
| `STALE_FRAMES`  | `15`            | Frames without detection before the lock is dropped        |
| `YOLO_MODEL`    | `yolov8n.pt`    | Path/name of weights file                                  |

---

## 🔐 Security

- ✅ Credentials live **only in `.env`** — never committed (see `.gitignore`).
- ✅ `.env.example` ships placeholders — safe to commit.
- ⚠️ The camera SOAP endpoint uses `verify=False` for self-signed certs. Pin the cert if you deploy this beyond a lab.
- ⚠️ RTSP credentials are injected into the stream URL for OpenCV — don't print or log full URLs in production.

---

## 🛠️ Requirements

- **Python** 3.10+
- **NVIDIA GPU** with CUDA 12.x (CPU works but inference is much slower)
- **ONVIF-compliant** PTZ camera (Profile S / T) reachable over HTTPS
- **FFmpeg** (bundled in the Docker image; install via `apt`/`brew` locally)

---

## 🗺️ Roadmap

- [ ] Multi-object tracking with stable IDs (ByteTrack / BoT-SORT)
- [ ] PID controller for pan/tilt instead of bang-bang
- [ ] Web UI (replace OpenCV window with a browser stream)
- [ ] Recording / event clipping when target is detected
- [ ] Profile-aware presets (`person-mode`, `airplane-mode`, `drone-mode`)

---

## 🤝 Contributing

PRs welcome. Keep changes focused, leave the `.env.example` keys in sync with any new env var, and don't commit weights.

---

## 📜 License

MIT — see [LICENSE](LICENSE) (add one before publishing).

---

<p align="center"><sub>Built with caffeine, OpenCV, and a stubborn bottle that refused to stay centered.</sub></p>
