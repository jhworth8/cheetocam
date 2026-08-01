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
and its Git revision without changing the detection or alert cadence.

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

## Tests

```bash
python3 -m pytest -q test_caption_logic.py test_cheeto_id.py
python3 -m py_compile critter_detector.py live_stream.py caption_logic.py cheeto_id.py
```

Reviewed labels stay in the Cheeto Box database. Embeddings and derived pet
profiles stay on the Pi in `learned_profiles.npz`; no base model is retrained.
