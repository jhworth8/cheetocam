#!/usr/bin/env python3
import cv2
import os
import smtplib
from email.message import EmailMessage
import re
import time
import threading
from datetime import datetime
from pytz import timezone
import numpy as np
import requests
import base64
import logging
import PIL.Image
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client
from ultralytics import YOLO

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
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Pushover credentials - hardcoded as requested
PUSHOVER_USER_KEY = "ubr5phjr9wymwabf8sg1anmsj9a12k"
PUSHOVER_API_TOKEN = "aqgxkbzacmm4mi8fpu49duzwnorfjf"

ENABLE_GEMINI = int(os.getenv('ENABLE_GEMINI', '1'))
ENABLE_CAT_DETECTION = int(os.getenv('ENABLE_CAT_DETECTION', '1'))
ENABLE_SUPABASE_UPLOAD = int(os.getenv('ENABLE_SUPABASE_UPLOAD', '1'))

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

# Animal classes for detection and notifications
ANIMAL_CLASSES = ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# Use only animal classes for detection
DETECTION_CLASSES = ANIMAL_CLASSES
ENABLE_MULTI_CLASS_DETECTION = 1

COOLDOWN_DURATION = int(os.getenv('COOLDOWN_DURATION', '180'))
FRAME_DELAY = float(os.getenv('FRAME_DELAY', '0.2'))

# Moondream (local VLM via Ollama) configuration. Keep-alive controls how long
# the model stays resident in RAM after the last request — once it expires,
# Ollama unloads to free ~1.7 GB. With a 3-min cooldown and 5-min keep-alive,
# consecutive visits stay warm; quiet hours unload.
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
MOONDREAM_MODEL = os.getenv('MOONDREAM_MODEL', 'moondream')
MOONDREAM_KEEP_ALIVE = os.getenv('MOONDREAM_KEEP_ALIVE', '5m')
MOONDREAM_TIMEOUT = float(os.getenv('MOONDREAM_TIMEOUT', '60'))
# Extra grace period to wait for Moondream after the GIF burst finishes.
# If still no result by then, we fall back to Gemini.
MOONDREAM_GRACE_AFTER_GIF = float(os.getenv('MOONDREAM_GRACE_AFTER_GIF', '10'))
ENABLE_MOONDREAM = int(os.getenv('ENABLE_MOONDREAM', '1'))

# Initialize Gemini API if enabled
if ENABLE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', '')
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

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
    },
}

def get_notification_settings():
    now = time.time()
    if now - _settings_cache['fetched_at'] < SETTINGS_REFRESH_SECONDS:
        return _settings_cache['value']
    try:
        resp = supabase_client.table('notification_settings').select('*').eq('id', 1).limit(1).execute()
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
    """Send a Pushover notification.

    priority: -2 lowest .. 2 emergency. 0 = normal (respects quiet hours),
    1 = high (bypasses quiet hours, highlighted). Default 0 — only escalate
    for non-cat/unexpected detections.
    """
    if not get_notification_settings()['pushover_enabled']:
        logging.info("Pushover notifications disabled via settings — skipping.")
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

def get_gemini_response(image_path, prompt):
    if not ENABLE_GEMINI:
        return ""
    try:
        sample_file = PIL.Image.open(image_path)
        model = genai.GenerativeModel(model_name="gemini-3.1-flash-lite")
        response = model.generate_content([prompt, sample_file])
        logging.info("Gemini response received.")
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Could not generate description."

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

def get_moondream_description(image_path, detected_classes):
    """Call local Ollama (Moondream) for an image description. Returns '' on failure."""
    if not ENABLE_MOONDREAM:
        return ""
    try:
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        classes_str = ", ".join(detected_classes)
        prompt = (
            f"An object detector saw a {classes_str} in this image. "
            "Describe in one or two short sentences what the animal looks like "
            "(color, markings) and what it's doing (sitting, walking, eating, etc.). "
            "If you also see anything notable like other animals, weather, or unusual "
            "posture, mention it briefly."
        )
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MOONDREAM_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "keep_alive": MOONDREAM_KEEP_ALIVE,
            },
            timeout=MOONDREAM_TIMEOUT,
        )
        if not resp.ok:
            # Log only the status + a short snippet of the response body so a
            # large error payload can't flood the logs.
            logging.error(
                f"Moondream HTTP {resp.status_code}: {_safe_log_snippet(resp.text)}"
            )
            return ""
        text = (resp.json().get("response") or "").strip()
        if _looks_like_base64_blob(text):
            logging.warning(
                f"Moondream returned what looks like a base64 blob "
                f"({len(text)} chars) — discarding."
            )
            return ""
        if text:
            logging.info(f"Moondream description: {_safe_log_snippet(text)!r}")
        return text
    except Exception as e:
        logging.error(f"Moondream error: {_safe_log_snippet(e, 200)}")
        return ""

class _DescriptionThread(threading.Thread):
    """Background worker that fetches a Moondream description.

    Started right when YOLO fires so the LLM runs in parallel with the
    5-frame GIF burst; the main loop joins it after the burst completes.
    """
    def __init__(self, image_path, detected_classes):
        super().__init__(daemon=True)
        self.image_path = image_path
        self.detected_classes = detected_classes
        self.result = ""
        self.source = "none"

    def run(self):
        self.result = get_moondream_description(self.image_path, self.detected_classes)
        if self.result:
            self.source = "moondream"

def resolve_description(thread, image_path, detected_classes):
    """Wait for the Moondream thread (with a grace period after GIF capture),
    then fall back to Gemini if Moondream returned nothing.
    Returns (description, source)."""
    thread.join(timeout=MOONDREAM_GRACE_AFTER_GIF)
    if thread.result:
        return thread.result, "moondream"
    logging.info("Moondream didn't return in time — falling back to Gemini.")
    classes_str = ", ".join(detected_classes)
    gemini_prompt = (
        f"An object detector saw a {classes_str} in this image. "
        "In one or two short sentences, describe the animal's appearance and "
        "what they're doing."
    )
    gemini_text = get_gemini_response(image_path, gemini_prompt)
    if gemini_text and gemini_text != "Could not generate description.":
        return gemini_text, "gemini"
    return "", "none"

def build_confirmation_prompt(detected_classes):
    """Structured prompt that returns a parseable VISIBLE/DESCRIPTION block."""
    classes_str = ", ".join(detected_classes)
    return (
        "You are confirming whether an animal that an object detector saw is "
        "actually visible in this image.\n\n"
        f"The detector reported: {classes_str}.\n\n"
        "Respond in exactly this format with no extra commentary:\n"
        "VISIBLE: yes  (or: no)\n"
        "DESCRIPTION: One or two short sentences. If yes, describe the animal "
        "(color/markings), what it's doing (sitting, walking, eating, etc.), "
        "and anything notable (wet fur, other animals nearby, posture, etc.). "
        "If no, briefly say what the image actually shows."
    )

def parse_gemini_confirmation(response_text, detected_classes):
    """Return (confirmed: bool, description: str).

    Prefers the structured VISIBLE/DESCRIPTION block. Falls back to keyword
    matching when Gemini returns free-form text.
    """
    if not response_text:
        return False, ""
    text = response_text.strip()
    description = text

    visible_match = re.search(r"VISIBLE\s*:\s*(yes|no)\b", text, re.IGNORECASE)
    desc_match = re.search(r"DESCRIPTION\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if desc_match:
        description = desc_match.group(1).strip()

    if visible_match:
        return visible_match.group(1).lower() == "yes", description

    # Fallback: free-form text — use the previous keyword/negation heuristic.
    lowered = text.lower()
    negation_patterns = [
        r"\bdon'?t see\b",
        r"\bdo not see\b",
        r"\bcannot see\b",
        r"\bcan'?t see\b",
        r"\bis not visible\b",
        r"\bnot visible\b",
    ]
    if any(re.search(p, lowered) for p in negation_patterns):
        return False, description
    for cls in detected_classes:
        if re.search(r'\b' + re.escape(cls.lower()) + r'\b', lowered):
            return True, description
    return False, description

def upload_detection_to_supabase(timestamp, gemini_response, main_image_path, detected_classes=None, detectionTemp=None, detectionWeather=None, detectionIcon=None):
    if not ENABLE_SUPABASE_UPLOAD:
        return
    try:
        with open(main_image_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        epoch = int(time.time())
        
        detection_data = {
            'timestamp': timestamp,
            'epoch': epoch,
            'gemini_response': gemini_response,
            'main_image': image_base64,
            'detectiontemp': detectionTemp,
            'detectionweather': detectionWeather,
            'detectionicon': detectionIcon
        }
        response = supabase_client.table("detections").insert(detection_data).execute()
        logging.info("Detection uploaded to Supabase with response: %s", response)
    except Exception as e:
        logging.error(f"Error uploading to Supabase: {e}")

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

logging.info("Camera initialized. Beginning main loop...")

cooldown_end_time = 0.0

try:
    while True:
        start_loop = time.time()
        ret, frame = cap.read()
        if not ret:
            logging.error("Frame grab failed.")
            break

        current_time = time.time()

        if ENABLE_CAT_DETECTION and current_time >= cooldown_end_time:
            # Choose detection method based on available model
            if model is not None:
                detections = detect_objects_yolo11(frame)
            else:
                detections = detect_objects_yolov3(frame)
            
            if detections:
                detected_classes = [d['class'] for d in detections]
                logging.info(f"Detected {len(detections)} objects: {detected_classes}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_image_path = f'detection_{timestamp}.jpg'
                cv2.imwrite(full_image_path, frame)

                # Kick off description generation in the background so it runs
                # in parallel with the 20-second GIF burst. We trust YOLO and
                # always alert; description is purely additive.
                desc_thread = _DescriptionThread(full_image_path, detected_classes)
                desc_thread.start()

                # Capture the burst GIF (~20s of action).
                gif_path = f'detection_{timestamp}.gif'
                try:
                    capture_burst_gif(cap, frame, gif_path)
                except Exception as e:
                    logging.error(f"GIF capture failed: {e}")
                    gif_path = None

                # Wait for Moondream (with grace period); fall back to Gemini
                # if it hasn't returned anything by then.
                description, desc_source = resolve_description(desc_thread, full_image_path, detected_classes)
                logging.info(f"Description source: {desc_source}; text={description!r}")

                # Fetch weather data
                temp, weather, icon = fetch_weather_data()

                # Subject + body
                eastern_now = datetime.now(timezone('US/Eastern'))
                time_str = eastern_now.strftime('%-I:%M %p ET')
                if 'cat' in detected_classes:
                    subject = "Cheeto's home 🐾"
                elif len(detected_classes) == 1:
                    subject = f"{detected_classes[0].title()} spotted 🐾"
                else:
                    subject = f"Visitors: {', '.join(detected_classes)} 🐾"

                body_lines = [f"{time_str} · {', '.join(detected_classes)}"]
                if description:
                    body_lines.append("")
                    body_lines.append(description)
                if temp and weather:
                    body_lines.append("")
                    body_lines.append(f"Weather: {temp:.0f}°F, {weather}")
                message = "\n".join(body_lines)

                # Email gets both still + GIF; Pushover prefers the GIF.
                attachments = [full_image_path]
                if gif_path and os.path.exists(gif_path):
                    attachments.append(gif_path)
                pushover_attachment = gif_path if (gif_path and os.path.exists(gif_path)) else full_image_path

                settings = get_notification_settings()
                if 'cat' in detected_classes:
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

                send_pushover_notification(
                    message=message,
                    title=subject,
                    image_path=pushover_attachment,
                    priority=push_priority,
                )

                upload_detection_to_supabase(
                    timestamp, description, full_image_path,
                    detected_classes, detectionTemp=temp,
                    detectionWeather=weather, detectionIcon=icon,
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
    cap.release()
    logging.info("Camera released. Program terminated.")
