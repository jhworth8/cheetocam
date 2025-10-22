#!/usr/bin/env python3
import cv2
import os
import smtplib
import imaplib
import email
from email.message import EmailMessage
import time
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
PHONE_RECIPIENTS = [r.strip() for r in os.getenv('PHONE_RECIPIENTS', '').split(',') if r.strip()]
EMAIL_RECIPIENTS = [r.strip() for r in os.getenv('EMAIL_RECIPIENTS', '').split(',') if r.strip()]
BOTHER_EMAIL = os.getenv('BOTHER_EMAIL', '').strip()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Pushover credentials - hardcoded as requested
PUSHOVER_USER_KEY = "ubr5phjr9wymwabf8sg1anmsj9a12k"
PUSHOVER_API_TOKEN = "aqgxkbzacmm4mi8fpu49duzwnorfjf"

ENABLE_GEMINI = int(os.getenv('ENABLE_GEMINI', '1'))
ENABLE_EMAIL_RESPONSE = int(os.getenv('ENABLE_EMAIL_RESPONSE', '1'))
ENABLE_CAT_DETECTION = int(os.getenv('ENABLE_CAT_DETECTION', '1'))
ENABLE_EMAIL_CHECK = int(os.getenv('ENABLE_EMAIL_CHECK', '1'))
ENABLE_ALERT_SENDING = int(os.getenv('ENABLE_ALERT_SENDING', '1'))
ENABLE_SUPABASE_UPLOAD = int(os.getenv('ENABLE_SUPABASE_UPLOAD', '1'))

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

COOLDOWN_DURATION = int(os.getenv('COOLDOWN_DURATION', '30'))
FRAME_DELAY = float(os.getenv('FRAME_DELAY', '0.2'))
EMAIL_CHECK_INTERVAL = int(os.getenv('EMAIL_CHECK_INTERVAL', '5'))

IMAP_SERVER = 'imap.gmail.com'
IMAP_PORT = 993

# Initialize Gemini API if enabled
if ENABLE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', '')
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

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

def send_email_with_attachments(image_paths, subject, message, phone_recipients, email_recipients):
    if not ENABLE_ALERT_SENDING:
        return
    all_recipients = phone_recipients + email_recipients
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
                if ext not in ['jpg', 'jpeg', 'png']:
                    ext = 'jpeg'
                msg.add_attachment(img_data, maintype='image', subtype=ext, filename=os.path.basename(image_path))
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
                logging.info(f"Email/MMS sent to {recipient} | Subject: {subject}")
        except Exception as e:
            logging.error(f"Failed to send to {recipient}: {e}")

def send_pushover_notification(message, title="Detection Alert", image_path=None):
    """Send a Pushover notification."""
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
                "priority": 1
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
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content([prompt, sample_file])
        logging.info("Gemini response received.")
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Could not generate description."

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

def check_email(cap):
    if not ENABLE_EMAIL_CHECK:
        return
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(SENDER_EMAIL, SENDER_PASSWORD)
        mail.select('inbox')
        result, data = mail.search(None, '(UNSEEN)')
        if result == 'OK':
            email_ids = data[0].split()
            for email_id in email_ids:
                result, msg_data = mail.fetch(email_id, '(RFC822)')
                if result != 'OK':
                    continue
                msg = email.message_from_bytes(msg_data[0][1])
                sender = email.utils.parseaddr(msg['From'])[1]
                logging.info(f"New email from: {sender}")
                all_known_senders = PHONE_RECIPIENTS + EMAIL_RECIPIENTS
                if sender not in all_known_senders:
                    mail.store(email_id, '+FLAGS', '\\Seen')
                    continue
                if ENABLE_EMAIL_RESPONSE:
                    ret, frame = cap.read()
                    if ret:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = f'email_triggered_{timestamp}.jpg'
                        cv2.imwrite(image_path, frame)
                        logging.info(f"Captured image: {image_path}")
                        gemini_response = ""
                        if ENABLE_GEMINI:
                            prompt = ("This image was captured in response to your inquiry. "
                                      "Please provide a clear and concise description of what you see. Use short sentences.")
                            gemini_response = get_gemini_response(image_path, prompt)
                        temp, weather, icon = fetch_weather_data()
                        eastern = timezone('US/Eastern')
                        subject = "Here's what's going on!"
                        message = "Currently outside the door...\n"
                        if gemini_response:
                            message += f"\nGemini Response:\n{gemini_response}"
                        if temp and weather:
                            message += f"\nWeather: {temp} °F, {weather}"
                        send_email_with_attachments(
                            image_paths=[image_path],
                            subject=subject,
                            message=message,
                            phone_recipients=[],
                            email_recipients=[sender]
                        )
                        send_pushover_notification(
                            message=message,
                            title="Email Triggered Detection",
                            image_path=image_path
                        )
                        upload_detection_to_supabase(timestamp, gemini_response, image_path, detectionTemp=temp, detectionWeather=weather, detectionIcon=icon)
                mail.store(email_id, '+FLAGS', '\\Seen')
        mail.logout()
    except Exception as e:
        logging.error(f"Email check error: {e}")

# Initialize camera
cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    logging.error("Cannot open camera")
    exit()

logging.info("Camera initialized. Beginning main loop...")

cooldown_end_time = 0.0
last_email_check_time = 0.0

try:
    while True:
        start_loop = time.time()
        ret, frame = cap.read()
        if not ret:
            logging.error("Frame grab failed.")
            break

        current_time = time.time()
        if (current_time - last_email_check_time >= EMAIL_CHECK_INTERVAL):
            check_email(cap)
            last_email_check_time = current_time

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

                gemini_response = ""
                if ENABLE_GEMINI:
                    prompt = ("Please provide a clear and concise description of the scene captured. "
                              "Use short sentences to describe what you see.")
                    gemini_response = get_gemini_response(full_image_path, prompt)

                # Fetch weather data when a detection is captured
                temp, weather, icon = fetch_weather_data()

                # Check if Gemini confirms the detection (any detected class, not just cat)
                logging.info(f"Gemini response: {gemini_response}")
                logging.info(f"Detected classes: {detected_classes}")
                
                # Ensure Gemini response actually mentions the detected animal
                gemini_confirmed = False
                gemini_lower = gemini_response.lower()
                
                # Check for negation phrases that would indicate Gemini doesn't see the animal
                negation_phrases = ["don't see", "do not see", "no ", "cannot see", "can't see", "is not visible", "not visible"]
                has_negation = any(phrase in gemini_lower for phrase in negation_phrases)
                
                if not has_negation:
                    for cls in detected_classes:
                        if cls.lower() in gemini_lower:
                            gemini_confirmed = True
                            logging.info(f"Gemini confirmed detection of {cls}")
                            break
                else:
                    logging.info("Gemini response contains negation - detection not confirmed")
                
                if gemini_confirmed:
                    logging.info("Gemini confirmed the detection. Sending alerts and uploading detection...")
                    
                    # Create subject based on detected animals
                    if len(detected_classes) == 1:
                        subject = f"🐾 {detected_classes[0].title()} Detected on {datetime.now(timezone('US/Eastern')).strftime('%B %d, %Y at %I:%M %p ET')}"
                    else:
                        subject = f"🐾 Multiple Animals Detected on {datetime.now(timezone('US/Eastern')).strftime('%B %d, %Y at %I:%M %p ET')}"
                    
                    # Create message for animal detection
                    message = f"🐾 Animal Detection Alert! 🐾\n\nDetected Animals: {', '.join(detected_classes)}\n"
                    
                    message += f"\n{gemini_response}"
                    if temp and weather:
                        message += f"\nWeather: {temp} °F, {weather}"
                    
                    # Send email notification with attachments
                    # Determine recipients based on detected animals
                    if 'cat' in detected_classes:
                        # Cat detection - send to normal recipients
                        send_email_with_attachments(
                            image_paths=[full_image_path],
                            subject=subject,
                            message=message,
                            phone_recipients=PHONE_RECIPIENTS,
                            email_recipients=EMAIL_RECIPIENTS
                        )
                    else:
                        # Non-cat animal detection - send to bother email
                        if BOTHER_EMAIL:
                            send_email_with_attachments(
                                image_paths=[full_image_path],
                                subject=subject,
                                message=message,
                                phone_recipients=[],
                                email_recipients=[BOTHER_EMAIL]
                            )
                    # Send Pushover notification with image attachment
                    send_pushover_notification(
                        message=message,
                        title=subject,
                        image_path=full_image_path
                    )
                    # Upload detection to Supabase
                    upload_detection_to_supabase(timestamp, gemini_response, full_image_path, detected_classes, detectionTemp=temp, detectionWeather=weather, detectionIcon=icon)
                    cooldown_end_time = current_time + COOLDOWN_DURATION
                else:
                    logging.info("Gemini did not confirm the detection. No alerts sent.")

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
