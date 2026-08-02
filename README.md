# Critter Cam detector

The Raspberry Pi detector for Critter Cam. It reads one USB webcam, confirms
animal presence with YOLO, recognizes household pets with locally derived CLIP
profiles, writes activity descriptions with local Florence-2, uploads frames to
the self-hosted Cheeto Box, and serves authenticated live video/audio.

No cloud AI provider is supported. If Florence-2 is unavailable, detection and
pet recognition continue without an activity description.

## Runtime

- Entry point: `critter_detector.py`
- Service: `cat_detector.service`
- Camera: `/dev/video0`
- Live endpoints: `/snapshot.jpg`, `/stream.mjpg`, `/audio.mp3`, `/health`
- Configuration: `.env` copied from `.env.example`
- Database access: one detector/service key and one explicit `HOUSEHOLD_ID`

The detector sends a heartbeat every 30 seconds. It reports the last frame,
last detection, local-model readiness, live-view readiness, audio capability,
and its Git revision without changing the detection or alert cadence. While a
visitor remains present, it can save another observation every
`OBSERVATION_INTERVAL_SECONDS` (60 seconds by default). Notification cooldown
and the once-per-visit policy are evaluated separately, so those extra frames
do not create extra alerts.

## Install

```bash
python3 -m pip install -r requirements.txt
sudo cp cat_detector.service /etc/systemd/system/cat_detector.service
sudo systemctl daemon-reload
sudo systemctl enable --now cat_detector.service
```

Before starting, copy `.env.example` to `.env`, set the household UUID and
self-hosted API credentials, and confirm `STREAM_AUDIO_DEVICE` with
`arecord -l`.

## Additional and wireless cameras

Run one detector service instance per camera. A local webcam uses
`CAMERA_SOURCE=/dev/video0`; a wireless/IP camera uses its RTSP or HTTP/MJPEG
stream URL. Give every instance a stable `CAMERA_ID`, a friendly
`CAMERA_NAME`, its own `STREAM_PORT`, token, and public tunnel base. Wireless
URLs—including usernames and passwords—stay only in that instance's `.env`.

```dotenv
CAMERA_ID=back-yard
CAMERA_NAME="Back yard"
CAMERA_SOURCE=rtsp://camera-user:password@192.168.1.80/stream1
CAMERA_SOURCE_KIND=wireless
STREAM_PORT=8089
STREAM_PUBLIC_BASE=https://back-yard-cam.example.com
STREAM_AUDIO_ENABLED=0
```

Install the template once, copy the existing private configuration as a
starting point, then enable the camera by its id:

```bash
sudo cp cat_detector@.service /etc/systemd/system/cat_detector@.service
cp .env .env.back-yard
chmod 600 .env.back-yard
# Edit only .env.back-yard and replace the camera-specific values above.
sudo systemctl daemon-reload
sudo systemctl enable --now cat_detector@back-yard.service
```

Keep the existing `cat_detector.service` for the front-door webcam. The second
instance reads only `.env.back-yard`; both can share the detector code and
household credentials without sharing camera URLs or ports. The camera appears
automatically in the household camera list when its first heartbeat arrives.
Detection history and visits remain separated by camera.

For embedded cameras whose one-shot JPEG endpoint is more reliable than their
MJPEG framing, keep `CAMERA_SOURCE_KIND=wireless`, set
`CAMERA_CAPTURE_MODE=snapshot`, point `CAMERA_SOURCE` at the JPEG URL, and
optionally tune `CAMERA_SNAPSHOT_INTERVAL` (default `0.2` seconds). The detector
polls and decodes those frames through the same capture interface used by video
sources. Set `CAMERA_ROTATION` to `90`, `180`, or `270` when the camera module's
fixed orientation does not match its mounting position.

Battery saver, toggled per camera from the app, slows that same poll to
`CAMERA_SAVER_SNAPSHOT_INTERVAL` (default `1.0` seconds) while nothing is
happening, and restores the full rate the moment somebody opens live view or
the detector sees anything at all — confirmed or not — for the next
`CAMERA_SAVER_ACTIVE_SECONDS` (default `20`). It only applies to
`CAMERA_CAPTURE_MODE=snapshot`: a camera the Pi reads as a stream is
transmitting whether or not the Pi reads it, so there is no poll to slow, and
the detector logs a warning once rather than pretend otherwise. The device's
own sleep and keep-alive behaviour is deliberately untouched — see the power
bank note in `cheeto-infra/devices/README.md`.

Each service has equal systemd CPU weight and a 150% CPU ceiling, leaving one
core of headroom on a four-core Pi when two feeds are busy. Capture and
lightweight YOLO run per camera, while CLIP/Florence jobs use a shared local inference gate
(`AI_SHARED_SLOTS=1` by default). That prevents two cameras from saturating all
four Pi cores at once; a busy identity check may be skipped after three seconds,
but the observation is still saved and the caption worker queues safely. Camera
sources are opened before models load, so an offline wireless camera reports its
error and retries later without repeatedly loading local AI.

For a battery camera that deep-sleeps between PIR events, set
`CAMERA_MOTION_WAKE=1` plus `CAMERA_CONTROL_URL`,
`CAMERA_CONTROL_USERNAME`, and `CAMERA_CONTROL_PASSWORD`. The Pi then treats a
missing source as expected sleep, keeps the public Live endpoint available, and
claims the camera's ESPHome `Live Hold` switch when the app begins polling. The
firmware's timer wake determines worst-case live-start latency; ordinary motion
still begins capture immediately.

## Tests

```bash
python3 -m pytest -q test_alert_logic.py test_caption_logic.py test_cheeto_id.py test_inference_gate.py
python3 -m py_compile critter_detector.py live_stream.py alert_logic.py caption_logic.py cheeto_id.py inference_gate.py
```

Reviewed labels stay in the Cheeto Box database. Embeddings and derived pet
profiles stay on the Pi in `learned_profiles.npz`; no base model is retrained.
