#!/usr/bin/env python3
import cv2
import os
import smtplib
import subprocess
from email.message import EmailMessage
import re
import time
import threading
import warnings
from datetime import datetime
from pytz import timezone
import numpy as np
import requests
import base64
import io
import hashlib
import logging
import PIL.Image
from dotenv import load_dotenv
from supabase import create_client, Client
from ultralytics import YOLO

from caption_logic import (
    ANIMAL_CLASSES,
    should_suppress,
    resolve_reported_classes,
)
# Module level, NOT inside the ENABLE_CHEETO_ID block below: the caption crop
# needs this whether or not prototype ID is switched on. cheeto_id only pulls
# in numpy and PIL at import time (torch is loaded lazily inside functions),
# so this stays cheap.
from cheeto_id import crop_with_context

# Silence the noisy "num_beams=1 with early_stopping=True" warning that
# Florence-2's generation config triggers on every call. We use greedy
# decoding intentionally for speed; the warning is cosmetic.
warnings.filterwarnings(
    "ignore",
    message=".*num_beams.*early_stopping.*",
    category=UserWarning,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("cat_detector.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting enhanced cat detection system with YOLOv11, Supabase and Pushover integration...")

# Email and API configuration
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
# Env recipients are fallbacks used only when notification_settings has no entries
ENV_PHONE_RECIPIENTS = [r.strip() for r in os.getenv('PHONE_RECIPIENTS', '').split(',') if r.strip()]
ENV_EMAIL_RECIPIENTS = [r.strip() for r in os.getenv('EMAIL_RECIPIENTS', '').split(',') if r.strip()]
ENV_BOTHER_EMAIL = os.getenv('BOTHER_EMAIL', '').strip()
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Two notification paths run in parallel.
#
# 1. Pushover — the proven one, kept running until the self-hosted path has
#    earned trust. Credentials now come from .env (which is gitignored) rather
#    than being hardcoded here, where they sat in a public repo.
# 2. The self-hosted dispatcher on cheeto, which fans out to ntfy (Android, no
#    third party at all) and APNs (iOS, Apple only, our own key). This box holds
#    only a bearer token; the push credentials live on cheeto.
#
# Each can be switched off independently, so Pushover can be retired later by
# flipping one env var rather than editing code.
PUSHOVER_USER_KEY = os.getenv('PUSHOVER_USER_KEY', '')
PUSHOVER_API_TOKEN = os.getenv('PUSHOVER_API_TOKEN', '')
ENABLE_PUSHOVER = int(os.getenv('ENABLE_PUSHOVER', '1'))

PUSH_DISPATCH_URL = os.getenv('PUSH_DISPATCH_URL', 'http://192.168.86.113:8090/push')
PUSH_DISPATCH_TOKEN = os.getenv('PUSH_DISPATCH_TOKEN', '')
ENABLE_DISPATCH_PUSH = int(os.getenv('ENABLE_DISPATCH_PUSH', '1'))

ENABLE_CAT_DETECTION = int(os.getenv('ENABLE_CAT_DETECTION', '1'))
ENABLE_SUPABASE_UPLOAD = int(os.getenv('ENABLE_SUPABASE_UPLOAD', '1'))

# Every detector belongs to exactly one household. The original installation
# keeps its stable default for a no-downtime upgrade; new installations receive
# their household UUID during camera setup.
DEFAULT_HOUSEHOLD_ID = '486f6c6c-696e-4773-8000-000000000001'
HOUSEHOLD_ID = os.getenv('HOUSEHOLD_ID', DEFAULT_HOUSEHOLD_ID).strip()
CAMERA_ID = os.getenv('CAMERA_ID', 'front-door').strip() or 'front-door'
CAMERA_NAME = os.getenv('CAMERA_NAME', 'Front door').strip() or 'Front door'
HEARTBEAT_SECONDS = max(15, int(os.getenv('HEARTBEAT_SECONDS', '30')))
PROCESS_STARTED_EPOCH = int(time.time())

# How often to re-fetch notification_settings from Supabase
SETTINGS_REFRESH_SECONDS = int(os.getenv('SETTINGS_REFRESH_SECONDS', '30'))

# Detection configuration
DETECTION_CONFIDENCE = float(os.getenv('DETECTION_CONFIDENCE', '0.7'))

# All 80 YOLOv11 classes for comprehensive detection
ALL_YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Animal classes for detection. Split into plausible vs lookalike classes in
# caption_logic.py: cow/elephant/giraffe/etc are still DETECTED (the nano
# model regularly mislabels the cat as one of them, and we don't want a
# mislabeled cat to slip through), but they're never REPORTED verbatim — the
# VLM caption decides what to call the animal.
DETECTION_CLASSES = ANIMAL_CLASSES
ENABLE_MULTI_CLASS_DETECTION = 1

# Require a target class in this many CONSECUTIVE frames before treating it
# as a real detection. Frames are ~FRAME_DELAY apart, so 2 adds ~0.2-0.4s of
# latency and kills single-frame YOLO flickers — the main false-positive
# source (shadows, the doormat pattern, headlights).
CONFIRM_FRAMES = int(os.getenv('CONFIRM_FRAMES', '2'))

COOLDOWN_DURATION = int(os.getenv('COOLDOWN_DURATION', '180'))
FRAME_DELAY = float(os.getenv('FRAME_DELAY', '0.2'))

# Florence-2 (local VLM via HuggingFace transformers) configuration.
# Replaces the Moondream/Ollama stack — Florence-2 base is ~230M params,
# loads in ~500 MB RAM, and produces a 1-2 sentence caption in ~15-20s
# on Pi 5 (vs Moondream's 70-85s). Much faster, somewhat dumber, no
# LLM-generated title — subject reverts to the generic "Cat at the door".
FLORENCE_MODEL_ID = os.getenv('FLORENCE_MODEL_ID', 'microsoft/Florence-2-base')
# Florence task prompt: "<CAPTION>" (short, ~5 words),
# "<DETAILED_CAPTION>" (one sentence), "<MORE_DETAILED_CAPTION>" (verbose).
FLORENCE_TASK = os.getenv('FLORENCE_TASK', '<DETAILED_CAPTION>')
FLORENCE_MAX_NEW_TOKENS = int(os.getenv('FLORENCE_MAX_NEW_TOKENS', '128'))
FLORENCE_NUM_BEAMS = int(os.getenv('FLORENCE_NUM_BEAMS', '1'))  # 1 = greedy (fastest)
# Extra grace period to wait for Florence after the GIF burst finishes.
# If it has not returned by then, proceed without an activity description.
FLORENCE_GRACE_AFTER_GIF = float(os.getenv('FLORENCE_GRACE_AFTER_GIF', '30'))
# How long to wait for Florence when the GIF burst is SKIPPED. The burst is
# what normally gives Florence its headroom: ~25s of capture plus the grace
# above, ~55s total. Drop the burst and the grace alone is the whole budget —
# and measured caption times are 27-33s, so a bare 30s would start timing out
# and losing the local description. Keep the total roughly the same.
FLORENCE_TIMEOUT_NO_GIF = float(os.getenv('FLORENCE_TIMEOUT_NO_GIF', '55'))
# Crop the frame around the animal before captioning. Florence was being
# handed the whole 640x480 porch, where the cat is a small fraction of the
# pixels — which is why captions ramble about flags, flower pots and clear
# blue skies, and why its species guess was so unreliable. Padding is
# deliberately generous: the point of the caption is what the cat is DOING,
# so it still needs to see the porch, just not be dominated by it.
FLORENCE_CROP_PAD_FRAC = float(os.getenv('FLORENCE_CROP_PAD_FRAC', '0.5'))
# Never hand Florence a postage stamp — expand the crop to at least this
# fraction of each frame dimension. A distant cat gives a tiny box, and a
# 40px crop upscaled to Florence's input is mush.
FLORENCE_CROP_MIN_FRAC = float(os.getenv('FLORENCE_CROP_MIN_FRAC', '0.4'))
ENABLE_FLORENCE = int(os.getenv('ENABLE_FLORENCE', '1'))

# Cheeto prototype (see train_cheeto_prototype.py). A CLIP vector averaged
# from our own library of this cat, which answers "is this Cheeto?" far better
# than any general model can — it's the only signal here trained on THIS cat.
# Off until a prototype file exists; without one the detector behaves exactly
# as it did before.
CHEETO_PROTOTYPE_PATH = os.getenv('CHEETO_PROTOTYPE_PATH', 'cheeto_prototype.npz')
ENABLE_CHEETO_ID = int(os.getenv('ENABLE_CHEETO_ID', '1'))
# CLIP shares the Pi's 4 cores with Florence. Leave one free so a detection
# doesn't starve the capture loop and drop frames.
CHEETO_ID_THREADS = int(os.getenv('CHEETO_ID_THREADS', '3'))
LEARNED_PROFILE_REFRESH_SECONDS = int(os.getenv('LEARNED_PROFILE_REFRESH_SECONDS', '300'))
LEARNED_PROFILE_THRESHOLD = float(os.getenv('LEARNED_PROFILE_THRESHOLD', '0.90'))
LEARNED_PROFILE_MIN_MARGIN = float(os.getenv('LEARNED_PROFILE_MIN_MARGIN', '0.015'))
LEARNED_PROFILE_CACHE_PATH = os.getenv(
    'LEARNED_PROFILE_CACHE_PATH', 'learned_profiles.npz')
# Drain this many frames immediately after a YOLO hit before saving the
# "detection frame", so USB auto-exposure has a moment to settle. Subsequent
# GIF frames are 5s later so they're always settled.
POST_DETECTION_SETTLE_FRAMES = int(os.getenv('POST_DETECTION_SETTLE_FRAMES', '10'))

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', '')
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

_last_frame_epoch = None
_last_detection_epoch = None
_heartbeat_error = None
_heartbeat_stop = threading.Event()


def detector_version():
    configured = os.getenv('DETECTOR_VERSION', '').strip()
    if configured:
        return configured
    try:
        return subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        return 'unknown'


DETECTOR_VERSION = detector_version()

# Notification settings cache. Refreshed from Supabase every
# SETTINGS_REFRESH_SECONDS so the dashboard can toggle channels remotely without
# restarting the service.
_settings_cache = {
    'fetched_at': 0.0,
    'value': {
        'email_enabled': True,
        'pushover_enabled': True,
        'email_recipients': ENV_EMAIL_RECIPIENTS,
        'phone_recipients': ENV_PHONE_RECIPIENTS,
        'bother_email': ENV_BOTHER_EMAIL,
        'cooldown_seconds': COOLDOWN_DURATION,
        # Preserve today's behaviour unless the owner explicitly chooses the
        # visit-aware notification mode in a newer client.
        'alert_mode': 'every_detection',
        'notification_title_style': 'smart',
        'notification_include_photo': True,
        'notification_include_activity': True,
        'notification_include_weather': True,
    },
}

def get_notification_settings():
    now = time.time()
    if now - _settings_cache['fetched_at'] < SETTINGS_REFRESH_SECONDS:
        return _settings_cache['value']
    try:
        resp = (supabase_client.table('notification_settings').select('*')
                .eq('household_id', HOUSEHOLD_ID).eq('id', 1)
                .limit(1).execute())
        rows = resp.data or []
        if rows:
            row = rows[0]
            email_recipients = row.get('email_recipients') or []
            phone_recipients = row.get('phone_recipients') or []
            bother_email = (row.get('bother_email') or '').strip()
            cooldown = row.get('cooldown_seconds')
            try:
                cooldown = int(cooldown) if cooldown is not None else COOLDOWN_DURATION
            except (TypeError, ValueError):
                cooldown = COOLDOWN_DURATION
            # Clamp to a sane range so a bad dashboard value can't break things.
            cooldown = max(10, min(cooldown, 3600))
            _settings_cache['value'] = {
                'email_enabled': bool(row.get('email_enabled', True)),
                'pushover_enabled': bool(row.get('pushover_enabled', True)),
                # Fall back to env recipients if the table column is empty so the
                # detector keeps working before the dashboard is populated.
                'email_recipients': email_recipients if email_recipients else ENV_EMAIL_RECIPIENTS,
                'phone_recipients': phone_recipients if phone_recipients else ENV_PHONE_RECIPIENTS,
                'bother_email': bother_email if bother_email else ENV_BOTHER_EMAIL,
                'cooldown_seconds': cooldown,
                'alert_mode': row.get('alert_mode') or 'every_detection',
                'notification_title_style': row.get('notification_title_style') or 'smart',
                'notification_include_photo': bool(row.get('notification_include_photo', True)),
                'notification_include_activity': bool(row.get('notification_include_activity', True)),
                'notification_include_weather': bool(row.get('notification_include_weather', True)),
            }
    except Exception as e:
        logging.error(f"Failed to fetch notification_settings, using last known values: {e}")
    _settings_cache['fetched_at'] = now
    return _settings_cache['value']

# Initialize YOLOv11 model
model = None
fallback_model = None
class_names = {}
target_classes = []

if ENABLE_CAT_DETECTION:
    try:
        # Try to load YOLOv11n model first
        logging.info("Loading YOLOv11n model...")
        model = YOLO('yolo11n.pt')
        logging.info("YOLOv11n model loaded successfully")
        
        # Get class names from the model
        class_names = model.names
        logging.info(f"YOLOv11 available classes: {len(class_names)}")
        
        # Filter detection classes to only those we care about
        if ENABLE_MULTI_CLASS_DETECTION:
            target_classes = []
            for cls in DETECTION_CLASSES:
                cls = cls.strip().lower()
                if cls in class_names.values():
                    target_classes.append(cls)
                else:
                    logging.warning(f"Class '{cls}' not found in YOLOv11 classes")
            
            if not target_classes:
                logging.warning("No valid detection classes found, defaulting to 'cat'")
                target_classes = ['cat']
            
            logging.info(f"Monitoring for classes: {target_classes}")
        else:
            target_classes = ['cat']
            logging.info("Monitoring for cats only")
            
    except Exception as e:
        logging.error(f"Failed to load YOLOv11 model: {e}")
        
        # Try fallback to YOLOv8n
        try:
            logging.info("Trying fallback to YOLOv8n...")
            model = YOLO('yolov8n.pt')
            class_names = model.names
            target_classes = ['cat', 'dog', 'person']  # Limited classes for fallback
            logging.info("YOLOv8n fallback model loaded successfully")
        except Exception as e2:
            logging.error(f"Failed to load fallback model: {e2}")
            
            # Final fallback to YOLOv3-tiny
            logging.info("Falling back to YOLOv3-tiny...")
            yolo_dir = os.getenv('YOLO_DIR', 'yolo')
            weights_path = os.path.join(yolo_dir, 'yolov3-tiny.weights')
            config_path = os.path.join(yolo_dir, 'yolov3-tiny.cfg')
            names_path = os.path.join(yolo_dir, 'coco.names')
            
            if os.path.exists(weights_path) and os.path.exists(config_path) and os.path.exists(names_path):
                with open(names_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                if 'cat' not in classes:
                    logging.error("'cat' class not found in COCO names.")
                    raise ValueError("'cat' class not found in COCO names.")
                cat_class_id = classes.index('cat')
                net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                layer_names = net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                model = None  # Flag to use old detection method
                target_classes = ['cat']
                logging.info("YOLOv3-tiny fallback loaded successfully")
            else:
                logging.error("No working YOLO model found!")
                raise Exception("No working YOLO model found!")

def fetch_weather_data():
    """Fetch current weather data from OpenWeatherMap."""
    try:
        lat = os.getenv('WEATHER_LAT', '42.5467')
        lon = os.getenv('WEATHER_LON', '-83.2113')
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=imperial"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        temp = data['main']['temp']
        weather = data['weather'][0]['main']
        icon = data['weather'][0]['icon']
        logging.info(f"Fetched weather: {temp} °F, {weather}")
        return temp, weather, icon
    except Exception as e:
        logging.error(f"Error fetching weather data: {e}")
        return None, None, None

def capture_burst_gif(cap, first_frame, gif_path, additional_frames=4, interval_s=5.0, gif_fps=2):
    """Capture extra fresh frames after a detection and write an animated GIF.

    Reads-and-discards buffered camera frames between captures so each saved
    frame reflects the current scene, not stale buffer contents. Frames are
    kept at the camera's native resolution (no resize, no crop).
    """
    frames = [first_frame]
    for _ in range(additional_frames):
        end_t = time.time() + interval_s
        last = None
        while time.time() < end_t:
            ret, f = cap.read()
            if ret:
                last = f
        if last is not None:
            frames.append(last)

    pil_frames = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_frames.append(PIL.Image.fromarray(rgb))

    duration_ms = int(1000 / gif_fps)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(gif_path) / 1024
    logging.info(f"Captured {len(frames)}-frame GIF -> {gif_path} ({size_kb:.0f} KB)")
    return gif_path

def send_email_with_attachments(image_paths, subject, message, phone_recipients, email_recipients):
    if not get_notification_settings()['email_enabled']:
        logging.info("Email notifications disabled via settings — skipping.")
        return
    all_recipients = phone_recipients + email_recipients
    if not all_recipients:
        logging.info("No email/phone recipients configured — skipping email send.")
        return
    for recipient in all_recipients:
        msg = EmailMessage()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.set_content(message)
        for image_path in image_paths:
            if not os.path.exists(image_path):
                logging.warning(f"Attachment {image_path} missing.")
                continue
            with open(image_path, 'rb') as img:
                img_data = img.read()
                _, ext = os.path.splitext(image_path)
                ext = ext.lower().replace('.', '')
                if ext == 'jpg':
                    ext = 'jpeg'
                if ext not in ['jpeg', 'png', 'gif']:
                    ext = 'jpeg'
                msg.add_attachment(img_data, maintype='image', subtype=ext, filename=os.path.basename(image_path))
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
                logging.info(f"Email/MMS sent to {recipient} | Subject: {subject}")
        except Exception as e:
            logging.error(f"Failed to send to {recipient}: {e}")

def send_pushover_notification(message, title="Detection Alert", image_path=None, priority=0):
    """Send a Pushover notification (legacy path, still the trusted one).

    image_path here is a LOCAL file — Pushover uploads the JPEG as a multipart
    attachment. That differs from send_push_notification below, which takes a
    Storage path instead. The local file still exists at call time; cleanup
    happens after both notifiers have run.

    priority: -2 lowest .. 2 emergency. 0 = normal (respects quiet hours),
    1 = high (bypasses quiet hours, highlighted).
    """
    if not ENABLE_PUSHOVER:
        return
    if not get_notification_settings()['pushover_enabled']:
        logging.info("Push notifications disabled via settings — skipping Pushover.")
        return
    if not (PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN):
        logging.error("Pushover credentials missing from .env — skipping Pushover.")
        return
    try:
        files = {}
        if image_path and os.path.exists(image_path):
            files['attachment'] = open(image_path, 'rb')
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_API_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "title": title,
                "message": message,
                "priority": priority,
            },
            files=files if files else None
        )
        if files:
            files['attachment'].close()
        if response.status_code == 200:
            logging.info("Pushover notification sent successfully.")
        else:
            logging.error("Failed to send Pushover notification: %s", response.text)
    except Exception as e:
        logging.error("Error sending Pushover notification: %s", e)

def send_push_notification(message, title="Detection Alert", image_path=None, priority=0):
    """Hand an alert to the self-hosted dispatcher on cheeto.

    image_path is a Storage path such as 'detections/thumb/20260730_160459.jpg',
    NOT a local file — the dispatcher turns it into a public URL that the phone
    fetches directly. The thumbnail is deliberately used rather than the full
    frame: it is about a fifth the bytes and renders identically in a
    notification shade or on a watch face.

    priority: 0 = normal, 1 = high. Kept identical to the old Pushover
    semantics so the call site and the settings toggle did not have to change.

    The settings flag is still called pushover_enabled — the column name is
    shared with the website and the app, so renaming it would break both for
    no functional gain.
    """
    if not ENABLE_DISPATCH_PUSH:
        return
    if not get_notification_settings()['pushover_enabled']:
        logging.info("Push notifications disabled via settings — skipping dispatcher.")
        return
    if not PUSH_DISPATCH_TOKEN:
        logging.error("PUSH_DISPATCH_TOKEN is unset — cannot send push.")
        return
    try:
        payload = {
            "title": title,
            "body": message,
            "priority": priority,
        }
        if image_path:
            payload["image_path"] = image_path
        response = requests.post(
            PUSH_DISPATCH_URL,
            json=payload,
            headers={"Authorization": f"Bearer {PUSH_DISPATCH_TOKEN}"},
            timeout=30,
        )
        if response.status_code == 200:
            logging.info("Push dispatched: %s", response.text[:200])
        else:
            logging.error("Push dispatcher returned %s: %s",
                          response.status_code, response.text[:200])
    except Exception as e:
        logging.error("Error dispatching push notification: %s", e)

# Florence-2 model + processor are loaded lazily on first use (or eagerly
# via load_florence_blocking() at startup) and kept resident for the
# lifetime of the process. Loading takes ~5-10s after model files are
# cached locally; first ever load downloads ~500 MB from HuggingFace.
_FLORENCE_MODEL = None
_FLORENCE_PROCESSOR = None

def load_florence_blocking(retry_seconds=30):
    """Load Florence-2 into memory. Blocks the main loop until it's ready
    — user explicitly does NOT want detections firing without the local
    VLM available. Retries indefinitely on transient failure (e.g. HF
    download flakiness)."""
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR
    if not ENABLE_FLORENCE:
        return
    if _FLORENCE_MODEL is not None:
        return
    attempt = 0
    while True:
        attempt += 1
        logging.info(
            f"Loading Florence-2 ({FLORENCE_MODEL_ID}) (attempt {attempt}). "
            "First run downloads ~500 MB."
        )
        t0 = time.time()
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            proc = AutoProcessor.from_pretrained(
                FLORENCE_MODEL_ID, trust_remote_code=True
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_ID, trust_remote_code=True
            ).eval()
            # Pi 5 CPU has no fp16 hardware — keep weights at fp32 for speed.
            mdl = mdl.to('cpu')
            _FLORENCE_PROCESSOR = proc
            _FLORENCE_MODEL = mdl
            logging.info(f"Florence-2 loaded in {time.time() - t0:.1f}s.")
            return
        except Exception as e:
            logging.warning(
                f"Florence-2 load failed: {_safe_log_snippet(e, 300)}. "
                f"Sleeping {retry_seconds}s before retrying. "
                "No detections will fire until Florence is loaded."
            )
            time.sleep(retry_seconds)

def _safe_log_snippet(s, limit=300):
    """Truncate a string for logging so we never dump a base64 image or other
    huge payload into journald."""
    if s is None:
        return ""
    s = str(s)
    if len(s) <= limit:
        return s
    return s[:limit] + f"...<{len(s) - limit} more chars>"

def _looks_like_base64_blob(s):
    """Heuristic: a 'description' that's almost entirely base64 chars and
    long is almost certainly the model echoing the image back."""
    if not s or len(s) < 200:
        return False
    sample = s[:500]
    b64_chars = sum(1 for c in sample if c.isalnum() or c in '+/=')
    return b64_chars / len(sample) > 0.95

def _downscaled_jpeg_b64(image_path, max_dim):
    """Load an image and re-encode as a smaller JPEG, returned base64.
    Used to send a lighter image to Moondream while keeping the original
    file intact for email/Pushover/Supabase."""
    img = PIL.Image.open(image_path).convert('RGB')
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def crop_for_caption(image, bbox):
    """Crop a frame around the animal for captioning. Thin wrapper binding
    the Florence-specific padding to the shared, unit-tested helper."""
    return crop_with_context(image, bbox, FLORENCE_CROP_PAD_FRAC,
                             FLORENCE_CROP_MIN_FRAC)


def get_florence_description(image, detected_classes):
    """Run Florence-2 on an already-loaded PIL image (normally the animal
    crop, see crop_for_caption). Returns the generated caption string, or ''
    on failure. Florence-2 doesn't follow free-form prompts — it takes a
    fixed task token (FLORENCE_TASK) and produces a caption."""
    if not ENABLE_FLORENCE or _FLORENCE_MODEL is None or _FLORENCE_PROCESSOR is None:
        return ""
    try:
        import torch
        image = image.convert('RGB')
        inputs = _FLORENCE_PROCESSOR(
            text=FLORENCE_TASK, images=image, return_tensors="pt"
        )
        with torch.inference_mode():
            generated_ids = _FLORENCE_MODEL.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=FLORENCE_MAX_NEW_TOKENS,
                num_beams=FLORENCE_NUM_BEAMS,
                do_sample=False,
            )
        raw = _FLORENCE_PROCESSOR.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = _FLORENCE_PROCESSOR.post_process_generation(
            raw, task=FLORENCE_TASK, image_size=(image.width, image.height)
        )
        # post_process_generation returns {TASK_TOKEN: "caption text..."}.
        caption = (parsed.get(FLORENCE_TASK) or "").strip()
        # Florence sometimes prefixes "The image shows..." — keep as-is, it's fine.
        if caption:
            logging.info(f"Florence-2 caption: {_safe_log_snippet(caption)!r}")
        return caption
    except Exception as e:
        logging.error(f"Florence-2 error: {_safe_log_snippet(e, 300)}")
        return ""

class _CaptionThread(threading.Thread):
    """Background worker that fetches a Florence-2 caption. Started right
    when YOLO fires so the VLM runs in parallel with the 5-frame GIF burst;
    the main loop joins it after the burst."""
    def __init__(self, image, detected_classes):
        super().__init__(daemon=True)
        self.image = image
        self.detected_classes = detected_classes
        self.description = ""
        self.started_at = time.time()
        self.finished_at = None

    def run(self):
        self.description = get_florence_description(
            self.image, self.detected_classes)
        self.finished_at = time.time()
        elapsed = self.finished_at - self.started_at
        if self.description:
            logging.info(f"Florence-2 completed in {elapsed:.1f}s.")
        else:
            logging.info(f"Florence-2 call finished in {elapsed:.1f}s with no usable result.")

def resolve_caption(thread, image_path, detected_classes, grace=None):
    """Wait for the local Florence-2 thread after GIF capture.

    If local captioning produces nothing, detection continues with no activity
    description. YOLO's multi-frame confirmation and CLIP pet recognition are
    independent of captioning, so a local-model failure never becomes a cloud
    call and never silently drops a real visitor.
    """
    # thread is None only when Florence is intentionally disabled for recovery.
    if thread is None:
        logging.info("Local captioning unavailable — continuing without a description.")
        return "", "none", False

    thread.join(timeout=FLORENCE_GRACE_AFTER_GIF if grace is None else grace)
    if thread.description:
        return thread.description, "florence", False
    logging.info("Florence-2 returned nothing — continuing without a description.")
    return "", "none", False

def upload_detection_to_supabase(
        timestamp, caption, main_image_path, detected_classes=None,
        detectionTemp=None, detectionWeather=None, detectionIcon=None,
        captured_epoch=None, identity_label=None, identity_confidence=None,
        identity_source=None, animal_id=None, bbox=None, needs_review=False,
        frame_quality=None):
    """Insert the detection row and put the image in Supabase Storage.

    Images used to be base64 stuffed into the row (~220KB of text each), which
    made the table 857MB and meant a phone had to download a full frame just to
    draw a grid thumbnail. They now go to the `detections` bucket as real files,
    with a 400px thumbnail alongside.

    Filenames are keyed on the detection timestamp rather than the row id, so
    the upload can happen before the insert and no second round trip is needed.
    The cooldown is minutes long, so same-second collisions are not a concern.

    Returns {'image_path', 'thumb_path'} on success, else None — the caller
    passes the thumbnail path to the push dispatcher.
    """
    if not ENABLE_SUPABASE_UPLOAD:
        return None
    try:
        name = f"{timestamp}.jpg"
        image_path = f"detections/{HOUSEHOLD_ID}/{name}"
        thumb_path = f"detections/{HOUSEHOLD_ID}/thumb/{name}"

        with open(main_image_path, 'rb') as f:
            raw = f.read()

        im = PIL.Image.open(io.BytesIO(raw)).convert('RGB')
        im.thumbnail((400, 400), PIL.Image.LANCZOS)
        tbuf = io.BytesIO()
        im.save(tbuf, format='JPEG', quality=80, optimize=True)

        headers = {
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
            'Content-Type': 'image/jpeg',
            'x-upsert': 'true',
        }
        for path, blob in ((image_path, raw), (thumb_path, tbuf.getvalue())):
            r = requests.post(f"{SUPABASE_URL}/storage/v1/object/{path}",
                              headers=headers, data=blob, timeout=60)
            if r.status_code not in (200, 201):
                raise RuntimeError(
                    f"storage upload {path} failed: {r.status_code} {r.text[:160]}")

        processed_epoch = int(time.time())
        legacy_data = {
            'household_id': HOUSEHOLD_ID,
            'timestamp': timestamp,
            # `epoch` remains the user-facing moment for old clients. It is the
            # capture instant now, not the time Florence happened to finish.
            'epoch': int(captured_epoch or processed_epoch),
            # Historical database column name; values are now generated only
            # by the local Florence model (or left blank).
            'gemini_response': caption,
            'image_path': image_path,
            'thumb_path': thumb_path,
            'detectiontemp': detectionTemp,
            'detectionweather': detectionWeather,
            'detectionicon': detectionIcon
        }
        rich_data = {
            **legacy_data,
            'captured_epoch': int(captured_epoch or processed_epoch),
            'processed_epoch': processed_epoch,
            'identity_label': identity_label,
            'identity_confidence': identity_confidence,
            'identity_source': identity_source,
            'animal_id': animal_id,
            'raw_classes': detected_classes or [],
            'model_version': 'yolo11n+clip+florence2-base',
            'bbox': bbox,
            'frame_quality': frame_quality,
            'needs_review': bool(needs_review),
        }
        try:
            response = supabase_client.table("detections").insert(rich_data).execute()
        except Exception as exc:
            # Deployment is deliberately rolling: the detector may update a
            # few minutes before the database migration. Keep recording rows
            # using the legacy contract instead of losing detections.
            logging.warning("Visit columns unavailable; using legacy insert: %s", exc)
            response = supabase_client.table("detections").insert(legacy_data).execute()
        inserted = len(getattr(response, 'data', None) or [])
        inserted_row = (getattr(response, 'data', None) or [{}])[0]
        visit_data = {}
        detection_id = inserted_row.get('id')
        if detection_id:
            try:
                visit_resp = supabase_client.rpc(
                    'attach_detection_to_visit',
                    {'p_detection_id': detection_id},
                ).execute()
                visit_data = getattr(visit_resp, 'data', None) or {}
            except Exception as exc:
                logging.warning("Visit attachment unavailable: %s", exc)
        logging.info(
            f"Detection uploaded ({inserted} row, image + thumb in Storage as {name}).")
        return {
            'image_path': image_path,
            'thumb_path': thumb_path,
            'detection_id': detection_id,
            **(visit_data if isinstance(visit_data, dict) else {}),
        }
    except Exception as e:
        logging.error(f"Error uploading to Supabase: {e}")
        return None


def update_detection_details(detection_id, description, temp, weather, icon,
                             identity_label, identity_confidence,
                             identity_source, needs_review):
    """Patch slow-caption results onto a row that may already have alerted.

    Confident Cheeto detections take the fast path so the photo reaches the
    phone in seconds. Florence still adds useful activity text afterwards, but
    it never causes a second notification.
    """
    if not detection_id:
        return
    legacy = {
        'gemini_response': description,
        'detectiontemp': temp,
        'detectionweather': weather,
        'detectionicon': icon,
    }
    rich = {
        **legacy,
        'processed_epoch': int(time.time()),
        'identity_label': identity_label,
        'identity_confidence': identity_confidence,
        'identity_source': identity_source,
        'needs_review': bool(needs_review),
    }
    try:
        (supabase_client.table('detections').update(rich)
         .eq('household_id', HOUSEHOLD_ID).eq('id', detection_id).execute())
    except Exception:
        try:
            (supabase_client.table('detections').update(legacy)
             .eq('household_id', HOUSEHOLD_ID).eq('id', detection_id).execute())
        except Exception as exc:
            logging.error("Could not update slow detection details: %s", exc)


def should_alert_for_visit(settings, storage):
    """Apply the optional visit notification mode.

    `every_detection` is the default and exactly preserves the current cadence.
    The visit-aware alternative exists but is inert until explicitly selected.
    """
    if settings.get('alert_mode', 'every_detection') != 'first_of_visit':
        return True
    return not storage or bool(storage.get('is_new_visit', True))

def detect_objects_yolo11(frame):
    """Detect objects using YOLOv11."""
    try:
        results = model(frame, conf=DETECTION_CONFIDENCE, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = class_names[class_id]
                    
                    if class_name in target_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        return detections
    except Exception as e:
        logging.error(f"YOLOv11 detection error: {e}")
        return []

def detect_objects_yolov3(frame):
    """Fallback detection using YOLOv3-tiny."""
    try:
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == cat_class_id and confidence > DETECTION_CONFIDENCE:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x1 = max(0, int(center_x - w / 2))
                    y1 = max(0, int(center_y - h / 2))
                    x2 = min(x1 + w, width)
                    y2 = min(y1 + h, height)
                    
                    detections.append({
                        'class': 'cat',
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return detections
    except Exception as e:
        logging.error(f"YOLOv3 detection error: {e}")
        return []

# Initialize camera
cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    logging.error("Cannot open camera")
    exit()

logging.info("Camera initialized.")

# On-demand live view. /dev/video0 only allows one reader and this process owns
# it, so the stream re-serves the frames already being decoded for YOLO rather
# than opening the camera a second time. Import and start are both guarded:
# a live view problem must never stop detection.
try:
    import live_stream
    live_stream.start(port=int(os.getenv('STREAM_PORT', '8088')),
                      token=os.getenv('STREAM_TOKEN', ''))
except Exception as _exc:
    live_stream = None
    logging.error("Live view unavailable: %s", _exc)

# Block until Florence-2 is loaded — the detector should never alert without
# the local VLM ready, per explicit configuration.
load_florence_blocking()


def send_camera_heartbeat(state=None, error_message=None):
    """Publish a tiny liveness row without touching detection cadence."""
    global _heartbeat_error
    now = int(time.time())
    local_ready = bool(ENABLE_FLORENCE and _FLORENCE_MODEL is not None)
    live_ready = bool(live_stream is not None and live_stream.ready())
    frame_fresh = bool(_last_frame_epoch and now - _last_frame_epoch <= 15)
    resolved_state = state or (
        'ready' if local_ready and live_ready and frame_fresh else 'degraded')
    resolved_error = error_message or _heartbeat_error
    payload = {
        'household_id': HOUSEHOLD_ID,
        'camera_id': CAMERA_ID,
        'display_name': CAMERA_NAME,
        'detector_version': DETECTOR_VERSION,
        'state': resolved_state,
        'process_started_epoch': PROCESS_STARTED_EPOCH,
        'heartbeat_epoch': now,
        'last_frame_epoch': _last_frame_epoch,
        'last_detection_epoch': _last_detection_epoch,
        'local_ai_ready': local_ready,
        'live_ready': live_ready,
        'audio_available': bool(
            live_stream is not None and live_stream.audio_available()),
        'error_message': resolved_error,
        'metrics': {
            'viewers': live_stream.viewers() if live_stream is not None else 0,
            'model': FLORENCE_MODEL_ID if local_ready else None,
        },
        'updated_at': datetime.now(timezone('UTC')).isoformat(),
    }
    try:
        (supabase_client.table('camera_status')
         .upsert(payload, on_conflict='household_id,camera_id').execute())
        _heartbeat_error = None
        return True
    except Exception as exc:
        _heartbeat_error = str(exc).splitlines()[0][:240]
        logging.warning("Camera heartbeat unavailable: %s", _heartbeat_error)
        return False


def heartbeat_loop():
    while not _heartbeat_stop.is_set():
        send_camera_heartbeat()
        _heartbeat_stop.wait(HEARTBEAT_SECONDS)


threading.Thread(
    target=heartbeat_loop,
    daemon=True,
    name='camera-heartbeat',
).start()

# Load the Cheeto prototype. A missing file is not an error — it just means
# the prototype has not been trained yet and species labeling uses the local
# caption plus conservative uncertainty rules.
_CHEETO_PROTOTYPE = None
identify_cheeto = None
if ENABLE_CHEETO_ID:
    try:
        from cheeto_id import (
            load_prototype,
            identify as identify_cheeto,
            crop_animal,
            embed_images,
            match_learned_profiles,
        )
        _CHEETO_PROTOTYPE = load_prototype(CHEETO_PROTOTYPE_PATH)
        if _CHEETO_PROTOTYPE:
            logging.info(
                f"Cheeto prototype loaded from {CHEETO_PROTOTYPE_PATH} "
                f"({_CHEETO_PROTOTYPE['n_images']} training crops, "
                f"threshold {_CHEETO_PROTOTYPE['threshold']:.3f})."
            )
        else:
            logging.info(
                f"No Cheeto prototype at {CHEETO_PROTOTYPE_PATH} — species ID "
                "falls back to the caption. Run train_cheeto_prototype.py."
            )
    except Exception as e:
        logging.error(f"Cheeto ID unavailable: {e}")


def load_learned_profile_cache(path=LEARNED_PROFILE_CACHE_PATH):
    """Load derived review prototypes without pickle or model weights."""
    if not path or not os.path.exists(path):
        return [], None
    try:
        data = np.load(path, allow_pickle=False)
        vectors = data['prototypes'].astype(np.float32)
        exemplar_matrix = data['exemplars'] if 'exemplars' in data.files else None
        exemplar_counts = (data['exemplar_counts']
                           if 'exemplar_counts' in data.files else None)
        profiles = []
        for i in range(len(vectors)):
            vector = vectors[i]
            norm = np.linalg.norm(vector)
            if not norm:
                continue
            profile = {
                'animal_id': int(data['animal_ids'][i]),
                'name': str(data['names'][i]),
                'slug': str(data['slugs'][i]),
                'species': str(data['species'][i]),
                'prototype': vector / norm,
                'threshold': float(data['thresholds'][i]),
                'pad_frac': 0.15,
                'model_name': 'ViT-B-32-quickgelu',
                'pretrained': 'openai',
                'n_images': int(data['image_counts'][i]),
            }
            if exemplar_matrix is not None and exemplar_counts is not None:
                profile['exemplars'] = exemplar_matrix[
                    i, :int(exemplar_counts[i])].astype(np.float32)
            profiles.append(profile)
        signature = str(data['sample_signature'])
        logging.info("Loaded %d reviewed visitor profile(s) from %s.",
                     len(profiles), path)
        return profiles, signature
    except Exception as exc:
        logging.warning("Could not load reviewed visitor cache %s: %s",
                        path, exc)
        return [], None


def save_learned_profile_cache(profiles, signature,
                               path=LEARNED_PROFILE_CACHE_PATH):
    if not profiles or not path:
        return
    temp_path = f"{path}.new"
    try:
        max_exemplars = max(len(p.get('exemplars', [])) for p in profiles)
        embedding_size = len(profiles[0]['prototype'])
        exemplar_matrix = np.zeros(
            (len(profiles), max_exemplars, embedding_size), dtype=np.float32)
        exemplar_counts = np.zeros(len(profiles), dtype=np.int32)
        for i, profile in enumerate(profiles):
            exemplars = np.asarray(profile.get('exemplars', []), dtype=np.float32)
            exemplar_counts[i] = len(exemplars)
            if len(exemplars):
                exemplar_matrix[i, :len(exemplars)] = exemplars
        with open(temp_path, 'wb') as handle:
            np.savez(
                handle,
                prototypes=np.stack([p['prototype'] for p in profiles]),
                animal_ids=np.asarray([p['animal_id'] for p in profiles]),
                names=np.asarray([p['name'] for p in profiles]),
                slugs=np.asarray([p['slug'] for p in profiles]),
                species=np.asarray([p['species'] for p in profiles]),
                thresholds=np.asarray([p['threshold'] for p in profiles]),
                image_counts=np.asarray([p['n_images'] for p in profiles]),
                exemplars=exemplar_matrix,
                exemplar_counts=exemplar_counts,
                sample_signature=np.asarray(signature),
            )
        os.replace(temp_path, path)
    except Exception as exc:
        logging.warning("Could not save reviewed visitor cache %s: %s",
                        path, exc)


_cached_profiles, _cached_signature = load_learned_profile_cache()
_learned_profile_cache = {
    'fetched_at': 0.0,
    'profiles': _cached_profiles,
    'signature': _cached_signature,
}


def get_learned_profiles(force=False):
    """Build conservative local prototypes from owner-reviewed visit samples.

    The database holds only sample references. Embeddings and prototypes stay
    on the Pi, are refreshed periodically, and reuse the already-loaded CLIP
    encoder. Profiles with fewer than three usable crops never participate.
    """
    if not ENABLE_CHEETO_ID or 'embed_images' not in globals():
        return []
    now = time.time()
    if (not force and now - _learned_profile_cache['fetched_at'] <
            LEARNED_PROFILE_REFRESH_SECONDS):
        return _learned_profile_cache['profiles']
    try:
        response = supabase_client.rpc(
            'profile_training_samples',
            {'p_household_id': HOUSEHOLD_ID},
        ).execute()
        rows = getattr(response, 'data', None) or []
        signature_source = 'multi-exemplar-v3|' + '|'.join(
            f"{row['animal_id']}:{row['detection_id']}"
            for row in sorted(rows,
                              key=lambda r: (r['animal_id'], r['detection_id']))
        )
        signature = hashlib.sha256(
            signature_source.encode('utf-8')).hexdigest()
        logging.info(
            "Reviewed visitor sample signature: %s (%d rows; cache %s).",
            signature[:10], len(rows),
            str(_learned_profile_cache.get('signature') or '')[:10] or 'none')
        if (_learned_profile_cache['profiles'] and
                signature == _learned_profile_cache.get('signature')):
            _learned_profile_cache['fetched_at'] = now
            logging.info("Reviewed visitor profiles are current; reused cache.")
            return _learned_profile_cache['profiles']

        grouped = {}
        for row in rows:
            grouped.setdefault(row['animal_id'], {
                'animal_id': row['animal_id'],
                'name': row['animal_name'],
                'slug': row['animal_slug'],
                'species': row.get('animal_species') or 'animal',
                'rows': [],
            })['rows'].append(row)

        profiles = []
        for profile in grouped.values():
            crops = []
            for row in profile['rows'][:24]:
                try:
                    image_response = requests.get(
                        f"{SUPABASE_URL}/storage/v1/object/public/{row['image_path']}",
                        timeout=20,
                    )
                    image_response.raise_for_status()
                    image = PIL.Image.open(io.BytesIO(image_response.content)).convert('RGB')
                    bbox = row.get('bbox')
                    if not bbox:
                        # Legacy/reviewed rows predate stored boxes. Recover
                        # the animal box locally from the full webcam frame so
                        # tapping "This is Cheeto" produces a real embedding
                        # instead of a sample the Pi silently throws away.
                        bgr = np.asarray(image)[:, :, ::-1].copy()
                        bbox = best_animal_bbox(bgr)
                    crop = crop_animal(image, bbox)
                    if crop is not None:
                        crops.append(crop)
                except Exception as exc:
                    logging.warning("Could not load learning sample %s: %s",
                                    row.get('detection_id'), exc)
            # One usable owner-confirmed crop may participate at a very strict
            # threshold. That is important for a brand-new visitor: excluding
            # Rocky entirely leaves only Cheeto in the comparison set, which
            # makes the classifier structurally incapable of choosing Rocky.
            if len(crops) < 1:
                continue
            vectors = embed_images(
                crops,
                model_name='ViT-B-32-quickgelu',
                pretrained='openai',
                num_threads=CHEETO_ID_THREADS,
            )
            # Remove the least centroid-consistent crops before learning. One
            # mislabeled empty mat must not drag Cheeto toward Rocky forever.
            rough = vectors.mean(axis=0)
            rough_norm = np.linalg.norm(rough)
            if not rough_norm:
                continue
            rough /= rough_norm
            consistency = vectors @ rough
            keep_at_least = min(3, len(vectors))
            keep_count = max(keep_at_least, int(np.ceil(len(vectors) * .80)))
            kept = vectors[np.argsort(consistency)[-keep_count:]]
            prototype = kept.mean(axis=0)
            norm = np.linalg.norm(prototype)
            if not norm:
                continue
            prototype = (prototype / norm).astype(np.float32)

            # Keep several genuinely different reviewed viewpoints. Start with
            # the most representative crop, then greedily choose the crop least
            # like the exemplars already kept.
            exemplar_indices = [int(np.argmax(kept @ prototype))]
            while len(exemplar_indices) < min(6, len(kept)):
                chosen = kept[exemplar_indices]
                redundancy = kept @ chosen.T
                candidate_order = np.argsort(np.max(redundancy, axis=1))
                next_index = next(
                    (int(i) for i in candidate_order if int(i) not in exemplar_indices),
                    None,
                )
                if next_index is None:
                    break
                exemplar_indices.append(next_index)

            profiles.append({
                'animal_id': profile['animal_id'],
                'name': profile['name'],
                'slug': profile['slug'],
                'species': profile['species'],
                'prototype': prototype,
                'exemplars': kept[exemplar_indices].astype(np.float32),
                'threshold': (max(LEARNED_PROFILE_THRESHOLD, .95)
                              if len(crops) < 2 else
                              (max(LEARNED_PROFILE_THRESHOLD, .925)
                               if len(crops) < 3
                               else LEARNED_PROFILE_THRESHOLD)),
                'pad_frac': 0.15,
                'model_name': 'ViT-B-32-quickgelu',
                'pretrained': 'openai',
                'n_images': len(crops),
            })
        _learned_profile_cache.update(
            fetched_at=now, profiles=profiles, signature=signature)
        save_learned_profile_cache(profiles, signature)
        logging.info("Learned visitor library refreshed: %d ready profile(s).",
                     len(profiles))
        return profiles
    except Exception as exc:
        # A rolling database deployment or temporary API error must not erase a
        # working in-memory library.
        logging.warning("Could not refresh learned visitor library: %s", exc)
        _learned_profile_cache['fetched_at'] = now
        return _learned_profile_cache['profiles']


def best_animal_bbox(frame):
    """Highest-confidence animal box in a frame, or None. Re-runs YOLO rather
    than reusing the trigger frame's box: the saved frame is captured
    POST_DETECTION_SETTLE_FRAMES later, by which point the cat has moved."""
    try:
        for det in sorted(detect_objects_yolo11(frame),
                          key=lambda d: d['confidence'], reverse=True):
            if det['class'] in ANIMAL_CLASSES:
                return det['bbox']
    except Exception as e:
        logging.error(f"Bounding box lookup failed: {e}")
    return None


def score_detection_frame(frame, bbox):
    """Fast 0..1 portrait quality used by the visit thumbnail selector."""
    if bbox is None or frame is None:
        return 0.0
    try:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        area_fraction = max(0, x2 - x1) * max(0, y2 - y1) / max(1, width * height)
        crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if crop.size == 0:
            return 0.0
        sharpness = cv2.Laplacian(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
        ).var()
        cx = (x1 + x2) / 2 / max(1, width)
        cy = (y1 + y2) / 2 / max(1, height)
        centrality = max(0.0, 1.0 - np.hypot(cx - .5, cy - .45) / .75)
        return float(np.clip(
            .55 * min(1.0, area_fraction / .28) +
            .25 * min(1.0, sharpness / 350.0) +
            .20 * centrality,
            0.0, 1.0,
        ))
    except Exception as exc:
        logging.debug("Could not score detection frame: %s", exc)
        return 0.0


def build_notification(reported_classes, hedged, correction_note, description,
                       temp, weather, captured_epoch, known_name=None,
                       settings=None):
    """Build user-facing copy from a stable identity decision.

    The time is when the webcam captured the frame, not when a slow caption
    finished. Cheeto gets his name; an uncertain weak-model species stays the
    deliberately generic "animal" until the owner labels the visit.
    """
    captured = datetime.fromtimestamp(captured_epoch, timezone('US/Eastern'))
    time_str = captured.strftime('%-I:%M %p ET')
    settings = settings or {}
    title_style = settings.get('notification_title_style', 'smart')
    if title_style == 'generic':
        subject = "Animal at the door 🐾"
    elif title_style == 'name_only':
        subject = f"{known_name or 'Animal'} at the door 🐾"
    elif known_name:
        subject = f"{known_name} at the door 🐾"
    elif len(reported_classes) == 1:
        label = reported_classes[0]
        subject = (f"Possible {label} at the door 🐾" if hedged
                   else f"{label.title()} at the door 🐾")
    else:
        joined = ', '.join(reported_classes)
        subject = (f"Possibly spotted: {joined} 🐾" if hedged
                   else f"Spotted: {joined} 🐾")

    body_lines = [f"{time_str} · {', '.join(reported_classes)}"]
    if correction_note:
        body_lines.append(correction_note)
    if description and settings.get('notification_include_activity', True):
        body_lines.extend(("", description))
    if (temp is not None and weather and
            settings.get('notification_include_weather', True)):
        body_lines.extend(("", f"Weather: {temp:.0f}°F, {weather}"))
    return subject, "\n".join(body_lines)

# Warm a changed review library once at startup. Subsequent five-minute checks
# compare a tiny sample-ID signature and reuse the on-disk prototype, so a
# normal detection never waits for 24 image downloads and CLIP embeddings.
get_learned_profiles(force=True)

logging.info("Beginning main loop...")

cooldown_end_time = 0.0
detection_streak = 0  # consecutive frames with a target class in view

try:
    while True:
        start_loop = time.time()
        ret, frame = cap.read()
        if not ret:
            logging.error("Frame grab failed.")
            _heartbeat_error = 'Camera frame grab failed'
            break

        _last_frame_epoch = int(time.time())

        # Hand the frame to the live view. Stores a reference only — no copy and
        # no JPEG encode — so this costs effectively nothing when nobody is
        # watching, and encoding happens on the serving thread when they are.
        if live_stream is not None:
            live_stream.publish(frame)

        current_time = time.time()

        if ENABLE_CAT_DETECTION and current_time >= cooldown_end_time:
            # Choose detection method based on available model
            if model is not None:
                detections = detect_objects_yolo11(frame)
            else:
                detections = detect_objects_yolov3(frame)

            if detections:
                detection_streak += 1
            else:
                detection_streak = 0

            if detections and detection_streak < CONFIRM_FRAMES:
                logging.info(
                    f"Possible {[d['class'] for d in detections]} — waiting for "
                    f"{CONFIRM_FRAMES} consecutive frames ({detection_streak}/{CONFIRM_FRAMES})."
                )
            elif detections:
                detection_streak = 0
                captured_epoch = int(current_time)
                _last_detection_epoch = captured_epoch
                detected_classes = [d['class'] for d in detections]
                logging.info(f"Detected {len(detections)} objects: {detected_classes}")

                # Drain a few frames so USB auto-exposure has a moment to settle.
                # The "detection moment" frame is often dark because the camera
                # was mid-adjustment when YOLO fired.
                settled_frame = frame
                for _ in range(POST_DETECTION_SETTLE_FRAMES):
                    ret, f = cap.read()
                    if ret:
                        settled_frame = f
                frame = settled_frame

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_image_path = f'detection_{timestamp}.jpg'
                cv2.imwrite(full_image_path, frame)

                # Kick off local caption generation in parallel with the GIF
                # burst, using the settled animal crop. Activity text never
                # leaves the Pi for processing.
                # One YOLO pass on the settled frame, shared by the caption
                # crop and the prototype. Re-run rather than reuse the trigger
                # frame's box: this frame is POST_DETECTION_SETTLE_FRAMES
                # later and the cat has usually moved since.
                animal_bbox = best_animal_bbox(frame)
                frame_quality = score_detection_frame(frame, animal_bbox)

                # Identity is the fast question and captioning is the slow one.
                # Compare every reviewed animal together. The old static Cheeto
                # prototype is only a fallback for households with no reviewed
                # library; letting it decide first is what caused Rocky to be
                # swallowed by a broad "orange cat" average.
                cheeto_verdict, cheeto_score = 'unknown', 0.0
                learned_profile = None
                learned_score = learned_margin = 0.0
                reviewed_profiles = get_learned_profiles()
                if reviewed_profiles and 'match_learned_profiles' in globals():
                    learned_profile, learned_score, learned_margin = match_learned_profiles(
                        full_image_path,
                        animal_bbox,
                        reviewed_profiles,
                        threshold=LEARNED_PROFILE_THRESHOLD,
                        min_margin=LEARNED_PROFILE_MIN_MARGIN,
                        num_threads=CHEETO_ID_THREADS,
                    )
                    if learned_profile:
                        logging.info(
                            "Reviewed visitor ID: %s (similarity %.3f, margin %.3f).",
                            learned_profile['name'], learned_score, learned_margin)
                        cheeto_verdict = ('cheeto'
                                          if learned_profile['slug'] == 'cheeto'
                                          else 'not_cheeto')
                        cheeto_score = learned_score
                elif _CHEETO_PROTOTYPE and identify_cheeto:
                    cheeto_verdict, cheeto_score = identify_cheeto(
                        full_image_path, animal_bbox, _CHEETO_PROTOTYPE,
                        num_threads=CHEETO_ID_THREADS,
                    )
                    logging.info(
                        f"Cheeto ID: {cheeto_verdict} (similarity "
                        f"{cheeto_score:.3f} vs threshold "
                        f"{_CHEETO_PROTOTYPE['threshold']:.3f})"
                    )

                known_name = (learned_profile['name'] if learned_profile else
                              ('Cheeto' if cheeto_verdict == 'cheeto' else None))
                known_slug = (learned_profile['slug'] if learned_profile else
                              ('cheeto' if cheeto_verdict == 'cheeto' else None))
                known_animal_id = (learned_profile['animal_id']
                                   if learned_profile else None)
                known_score = (learned_score if learned_profile else cheeto_score)
                known_source = ('clip-reviewed-exemplars' if learned_profile
                                else ('clip-prototype'
                                      if cheeto_verdict == 'cheeto' else None))

                early_settings = get_notification_settings()
                temp = weather = icon = None
                storage = None
                pushed = False

                if known_name:
                    temp, weather, icon = fetch_weather_data()
                    fast_description = f"{known_name} is at the door."
                    storage = upload_detection_to_supabase(
                        timestamp, fast_description, full_image_path,
                        detected_classes, detectionTemp=temp,
                        detectionWeather=weather, detectionIcon=icon,
                        captured_epoch=captured_epoch,
                        identity_label=known_slug,
                        identity_confidence=known_score,
                        identity_source=known_source,
                        animal_id=known_animal_id, bbox=animal_bbox,
                        needs_review=False,
                        frame_quality=frame_quality,
                    )
                    subject, fast_message = build_notification(
                        [('cat' if cheeto_verdict == 'cheeto' else
                          learned_profile.get('species', 'animal'))],
                        False, None, fast_description,
                        temp, weather, captured_epoch, known_name=known_name,
                        settings=early_settings)
                    if should_alert_for_visit(early_settings, storage):
                        include_photo = early_settings.get(
                            'notification_include_photo', True)
                        send_pushover_notification(
                            message=fast_message, title=subject,
                            image_path=(full_image_path if include_photo else None),
                            priority=0)
                        send_push_notification(
                            message=fast_message, title=subject,
                            image_path=((storage or {}).get('thumb_path')
                                        if include_photo else None),
                            priority=0)
                        pushed = True
                    logging.info("Fast-path known-visitor alert completed before captioning.")

                if ENABLE_FLORENCE:
                    caption_image = crop_for_caption(
                        PIL.Image.open(full_image_path).convert('RGB'),
                        animal_bbox)
                    caption_thread = _CaptionThread(caption_image, detected_classes)
                    caption_thread.start()
                else:
                    caption_thread = None

                # Capture the burst GIF (~25s of action) — but only when email
                # is on, because email is the ONLY channel it ever reaches.
                # Pushover deliberately gets the still (smartwatches render
                # GIFs as a blank frame). With email off we were spending 25s
                # of camera reads and CPU building a file we then deleted
                # unsent, starving Florence of cores on a 4-core Pi.
                gif_path = None
                if (early_settings['email_enabled'] and
                        early_settings.get('notification_include_photo', True)):
                    gif_path = f'detection_{timestamp}.gif'
                    try:
                        capture_burst_gif(cap, frame, gif_path)
                    except Exception as e:
                        logging.error(f"GIF capture failed: {e}")
                        gif_path = None
                else:
                    logging.info(
                        "Email disabled — skipping the GIF burst (it only ever "
                        "goes out by email)."
                    )

                # Wait only for local Florence-2. Without the burst there is no
                # 25s of incidental headroom, so extend the wait to keep the
                # total budget the same. If it still has no result, the visit
                # is saved and alerted without an activity description.
                description, caption_source, vlm_says_absent = resolve_caption(
                    caption_thread, full_image_path, detected_classes,
                    grace=None if gif_path else FLORENCE_TIMEOUT_NO_GIF)
                logging.info(
                    f"Caption source: {caption_source}; "
                    f"description={_safe_log_snippet(description)!r}"
                )

                # False-positive filter: skip everything (notification +
                # upload) only when the local caption describes an
                # animal-free scene.
                # Catches YOLO hallucinating a "dog" from shadows or the
                # doormat pattern.
                if (not known_name and
                        (vlm_says_absent or
                         should_suppress(detected_classes, description))):
                    logging.info(
                        "Suppressing alert: AI says no animal present. "
                        f"YOLO classes were {detected_classes}, "
                        f"description: {_safe_log_snippet(description)!r}"
                    )
                    # Clean up local files for the dropped detection.
                    for p in (full_image_path, gif_path):
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                    # Short cooldown so a flickering YOLO false-positive doesn't
                    # spam Florence on every frame, but not the full cooldown
                    # since this wasn't a real detection.
                    cooldown_end_time = current_time + 30
                    continue

                # Fetch weather data
                if temp is None:
                    temp, weather, icon = fetch_weather_data()

                # Report what the VLM caption says is there, not YOLO's raw
                # label — the nano model loves calling the cat a cow/elephant/
                # giraffe. With no caption, lookalike labels are hedged to cat.
                reported_classes, correction_note, hedged = resolve_reported_classes(
                    detected_classes, description, caption_source, cheeto_verdict)
                if reported_classes != detected_classes:
                    logging.info(
                        f"Reporting {reported_classes} (hedged={hedged}) — "
                        f"YOLO's raw labels were {detected_classes}."
                    )

                subject, message = build_notification(
                    reported_classes, hedged, correction_note, description,
                    temp, weather, captured_epoch,
                    known_name=known_name, settings=early_settings)

                # Email gets both still + GIF (GIF animates in mail clients).
                # Push gets the STILL JPEG thumbnail — smartwatches (Apple Watch,
                # Wear OS, Pixel Watch) reliably preview static images but
                # often render GIFs as a blank frame in their tiny notification
                # surface, killing the at-a-glance preview. That still now comes
                # from Storage rather than being attached to the request.
                attachments = ([full_image_path]
                               if early_settings.get('notification_include_photo', True)
                               else [])
                if (attachments and gif_path and os.path.exists(gif_path)):
                    attachments.append(gif_path)

                settings = get_notification_settings()
                if 'cat' in reported_classes:
                    send_email_with_attachments(
                        image_paths=attachments,
                        subject=subject,
                        message=message,
                        phone_recipients=settings['phone_recipients'],
                        email_recipients=settings['email_recipients'],
                    )
                    push_priority = 0
                else:
                    if settings['bother_email']:
                        send_email_with_attachments(
                            image_paths=attachments,
                            subject=subject,
                            message=message,
                            phone_recipients=[],
                            email_recipients=[settings['bother_email']],
                        )
                    push_priority = 1

                identity_label = (known_slug if known_slug
                                  else (reported_classes[0]
                                        if len(reported_classes) == 1
                                        else 'unknown'))
                identity_source = (known_source
                                   if known_source
                                   else caption_source)
                needs_review = (not known_name and
                                (reported_classes == ['animal'] or hedged))

                if storage:
                    update_detection_details(
                        storage.get('detection_id'), description, temp, weather,
                        icon, identity_label, known_score, identity_source,
                        needs_review)
                else:
                    # Upload runs before push because the notification fetches
                    # the thumbnail from Storage.
                    storage = upload_detection_to_supabase(
                        timestamp, description, full_image_path,
                        detected_classes, detectionTemp=temp,
                        detectionWeather=weather, detectionIcon=icon,
                        captured_epoch=captured_epoch,
                        identity_label=identity_label,
                        identity_confidence=known_score,
                        identity_source=identity_source,
                        animal_id=known_animal_id, bbox=animal_bbox,
                        needs_review=needs_review,
                        frame_quality=frame_quality,
                    )

                # Both notifiers fire while the self-hosted path is on trial, so
                # expect two alerts per detection until ENABLE_PUSHOVER is set
                # to 0. Pushover attaches the LOCAL jpeg; the dispatcher refers
                # to the thumbnail already in Storage.
                if not pushed and should_alert_for_visit(settings, storage):
                    include_photo = settings.get('notification_include_photo', True)
                    send_pushover_notification(
                        message=message,
                        title=subject,
                        image_path=(full_image_path if include_photo else None),
                        priority=push_priority,
                    )

                    send_push_notification(
                        message=message,
                        title=subject,
                        image_path=((storage or {}).get('thumb_path')
                                    if include_photo else None),
                        priority=push_priority,
                    )

                # Clean up local files so the Pi disk doesn't fill up.
                for p in (full_image_path, gif_path):
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError as e:
                            logging.warning(f"Could not remove {p}: {e}")

                cooldown_end_time = current_time + settings['cooldown_seconds']

        elapsed_time = time.time() - start_loop
        if elapsed_time < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed_time)

except KeyboardInterrupt:
    logging.info("Exiting...")

except Exception as e:
    logging.error(f"Error occurred: {e}")

finally:
    _heartbeat_stop.set()
    send_camera_heartbeat(state='error', error_message='Detector stopped')
    cap.release()
    logging.info("Camera released. Program terminated.")
