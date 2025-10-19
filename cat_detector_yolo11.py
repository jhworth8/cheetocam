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
from supabase import create_client, Client
from ultralytics import YOLO
import detector_config as config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG['level']),
    format=config.LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(config.LOGGING_CONFIG['file']),
        logging.StreamHandler() if config.LOGGING_CONFIG['console'] else logging.NullHandler()
    ]
)

logging.info("Starting enhanced cat detection system with YOLOv11 and comprehensive object detection...")

# Extract configuration
EMAIL_CONFIG = config.EMAIL_CONFIG
API_CONFIG = config.API_CONFIG
DETECTION_CONFIG = config.DETECTION_CONFIG
WEATHER_CONFIG = config.WEATHER_CONFIG
CAMERA_CONFIG = config.CAMERA_CONFIG
PERFORMANCE_CONFIG = config.PERFORMANCE_CONFIG
NOTIFICATION_CONFIG = config.NOTIFICATION_CONFIG
BACKUP_CONFIG = config.BACKUP_CONFIG
DEBUG_CONFIG = config.DEBUG_CONFIG

# Initialize Gemini API if enabled
if DETECTION_CONFIG['enable_gemini']:
    genai.configure(api_key=API_CONFIG['gemini_api_key'])

# Initialize Supabase client
if DETECTION_CONFIG['enable_supabase_upload']:
    supabase_client: Client = create_client(API_CONFIG['supabase_url'], API_CONFIG['supabase_anon_key'])

# Initialize YOLOv11 model
model = None
fallback_model = None
class_names = {}
target_classes = []

if DETECTION_CONFIG['enable_cat_detection']:
    try:
        # Try to load YOLOv11n model first
        logging.info(f"Loading YOLOv11 model: {DETECTION_CONFIG['model_name']}")
        model = YOLO(DETECTION_CONFIG['model_name'])
        logging.info("YOLOv11n model loaded successfully")
        
        # Get class names from the model
        class_names = model.names
        logging.info(f"YOLOv11 available classes: {len(class_names)}")
        
        # Filter detection classes to only those we care about
        if DETECTION_CONFIG['enable_multi_class_detection']:
            target_classes = []
            for cls in DETECTION_CONFIG['detection_classes']:
                cls = cls.strip().lower()
                if cls in class_names.values():
                    target_classes.append(cls)
                else:
                    if DEBUG_CONFIG['log_detection_details']:
                        logging.warning(f"Class '{cls}' not found in YOLOv11 classes")
            
            # Limit classes for performance
            if len(target_classes) > PERFORMANCE_CONFIG['max_detection_classes']:
                # Prioritize important classes
                priority_classes = ['cat', 'dog', 'person', 'bird', 'car', 'truck']
                target_classes = [cls for cls in priority_classes if cls in target_classes]
                target_classes.extend([cls for cls in DETECTION_CONFIG['detection_classes'] 
                                    if cls not in priority_classes][:PERFORMANCE_CONFIG['max_detection_classes'] - len(target_classes)])
            
            logging.info(f"Monitoring for {len(target_classes)} classes: {target_classes}")
        else:
            target_classes = ['cat']
            logging.info("Monitoring for cats only")
            
    except Exception as e:
        logging.error(f"Failed to load YOLOv11 model: {e}")
        
        # Try fallback to YOLOv8n
        try:
            logging.info(f"Trying fallback model: {DETECTION_CONFIG['fallback_model']}")
            model = YOLO(DETECTION_CONFIG['fallback_model'])
            class_names = model.names
            target_classes = ['cat', 'dog', 'person']  # Limited classes for fallback
            logging.info("YOLOv8n fallback model loaded successfully")
        except Exception as e2:
            logging.error(f"Failed to load fallback model: {e2}")
            
            # Final fallback to YOLOv3-tiny
            logging.info("Falling back to YOLOv3-tiny...")
            yolo_dir = BACKUP_CONFIG['yolo_dir']
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
                target_classes = BACKUP_CONFIG['backup_detection_classes']
                logging.info("YOLOv3-tiny fallback loaded successfully")
            else:
                logging.error("No working YOLO model found!")
                raise Exception("No working YOLO model found!")

def fetch_weather_data():
    """Fetch current weather data from OpenWeatherMap."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={WEATHER_CONFIG['latitude']}&lon={WEATHER_CONFIG['longitude']}&appid={API_CONFIG['openweather_api_key']}&units=imperial"
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
    if not DETECTION_CONFIG['enable_alert_sending']:
        return
    all_recipients = phone_recipients + email_recipients
    for recipient in all_recipients:
        msg = EmailMessage()
        msg['From'] = EMAIL_CONFIG['sender_email']
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
                server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
                server.send_message(msg)
                logging.info(f"Email/MMS sent to {recipient} | Subject: {subject}")
        except Exception as e:
            logging.error(f"Failed to send to {recipient}: {e}")

def send_pushover_notification(message, title="Detection Alert", image_path=None):
    """Send a Pushover notification."""
    if not NOTIFICATION_CONFIG['enable_pushover']:
        return
    try:
        files = {}
        if image_path and os.path.exists(image_path):
            files['attachment'] = open(image_path, 'rb')
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": API_CONFIG['pushover_api_token'],
                "user": API_CONFIG['pushover_user_key'],
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
    if not DETECTION_CONFIG['enable_gemini']:
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
    if not DETECTION_CONFIG['enable_supabase_upload']:
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
            'detectionicon': detectionIcon,
            'detected_classes': ','.join(detected_classes) if detected_classes else 'cat'
        }
        response = supabase_client.table("detections").insert(detection_data).execute()
        logging.info("Detection uploaded to Supabase with response: %s", response)
    except Exception as e:
        logging.error(f"Error uploading to Supabase: {e}")

def detect_objects_yolo11(frame):
    """Detect objects using YOLOv11."""
    try:
        # Resize frame for performance if configured
        if PERFORMANCE_CONFIG['image_resize']:
            frame_resized = cv2.resize(frame, PERFORMANCE_CONFIG['image_resize'])
        else:
            frame_resized = frame
            
        results = model(frame_resized, conf=DETECTION_CONFIG['detection_confidence'], verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = class_names[class_id]
                    
                    if class_name in target_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Scale coordinates back to original frame size
                        if PERFORMANCE_CONFIG['image_resize']:
                            scale_x = frame.shape[1] / PERFORMANCE_CONFIG['image_resize'][0]
                            scale_y = frame.shape[0] / PERFORMANCE_CONFIG['image_resize'][1]
                            x1, x2 = x1 * scale_x, x2 * scale_x
                            y1, y2 = y1 * scale_y, y2 * scale_y
                        
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
                
                if class_id == cat_class_id and confidence > DETECTION_CONFIG['detection_confidence']:
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
    if not DETECTION_CONFIG['enable_email_check']:
        return
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        mail.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
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
                all_known_senders = EMAIL_CONFIG['phone_recipients'] + EMAIL_CONFIG['email_recipients']
                if sender not in all_known_senders:
                    mail.store(email_id, '+FLAGS', '\\Seen')
                    continue
                if DETECTION_CONFIG['enable_email_response']:
                    ret, frame = cap.read()
                    if ret:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = f'email_triggered_{timestamp}.jpg'
                        cv2.imwrite(image_path, frame)
                        logging.info(f"Captured image: {image_path}")
                        gemini_response = ""
                        if DETECTION_CONFIG['enable_gemini']:
                            prompt = ("This image was captured in response to your inquiry. "
                                      "Please provide a clear and concise description of what you see. Use short sentences.")
                            gemini_response = get_gemini_response(image_path, prompt)
                        temp, weather, icon = fetch_weather_data()
                        eastern = timezone(WEATHER_CONFIG['timezone'])
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
cap = cv2.VideoCapture(CAMERA_CONFIG['device'])
if not cap.isOpened():
    logging.error("Cannot open camera")
    exit()

# Set camera resolution if specified
if CAMERA_CONFIG['resolution']:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['resolution'][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['resolution'][1])

logging.info("Camera initialized. Beginning main loop...")

cooldown_end_time = 0.0
last_email_check_time = 0.0
last_notification_time = {}

try:
    while True:
        start_loop = time.time()
        ret, frame = cap.read()
        if not ret:
            logging.error("Frame grab failed.")
            break

        current_time = time.time()
        if (current_time - last_email_check_time >= DETECTION_CONFIG['email_check_interval']):
            check_email(cap)
            last_email_check_time = current_time

        if DETECTION_CONFIG['enable_cat_detection'] and current_time >= cooldown_end_time:
            # Choose detection method based on available model
            if model is not None:
                detections = detect_objects_yolo11(frame)
            else:
                detections = detect_objects_yolov3(frame)
            
            if detections:
                detected_classes = [d['class'] for d in detections]
                logging.info(f"Detected {len(detections)} objects: {detected_classes}")
                
                # Check notification cooldown for each class
                should_notify = False
                for cls in detected_classes:
                    if cls not in last_notification_time or \
                       current_time - last_notification_time[cls] >= NOTIFICATION_CONFIG['notification_cooldown']:
                        should_notify = True
                        last_notification_time[cls] = current_time
                        break
                
                if should_notify:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    full_image_path = f'detection_{timestamp}.jpg'
                    cv2.imwrite(full_image_path, frame)

                    gemini_response = ""
                    if DETECTION_CONFIG['enable_gemini']:
                        prompt = ("Please provide a clear and concise description of the scene captured. "
                                  "Use short sentences to describe what you see.")
                        gemini_response = get_gemini_response(full_image_path, prompt)

                    # Fetch weather data when a detection is captured
                    temp, weather, icon = fetch_weather_data()

                    # Check if Gemini confirms the detection
                    if gemini_response and any(cls in gemini_response.lower() for cls in detected_classes):
                        logging.info("Gemini confirmed the detection. Sending alerts and uploading detection...")
                        
                        # Create subject based on detected classes
                        if len(detected_classes) == 1:
                            subject = f"{detected_classes[0].title()} Detected at {datetime.now(timezone(WEATHER_CONFIG['timezone'])).strftime('%I:%M %p ET')}"
                        else:
                            subject = f"Multiple Objects Detected at {datetime.now(timezone(WEATHER_CONFIG['timezone'])).strftime('%I:%M %p ET')}"
                        
                        message = f"Detection Alert!\n\nDetected: {', '.join(detected_classes)}\n\n" + gemini_response
                        if temp and weather:
                            message += f"\nWeather: {temp} °F, {weather}"
                        
                        # Send email notification with attachments
                        send_email_with_attachments(
                            image_paths=[full_image_path],
                            subject=subject,
                            message=message,
                            phone_recipients=EMAIL_CONFIG['phone_recipients'],
                            email_recipients=EMAIL_CONFIG['email_recipients']
                        )
                        # Send Pushover notification with image attachment
                        send_pushover_notification(
                            message=message,
                            title=subject,
                            image_path=full_image_path
                        )
                        # Upload detection to Supabase
                        upload_detection_to_supabase(timestamp, gemini_response, full_image_path, detected_classes, detectionTemp=temp, detectionWeather=weather, detectionIcon=icon)
                        cooldown_end_time = current_time + DETECTION_CONFIG['cooldown_duration']

        elapsed_time = time.time() - start_loop
        if elapsed_time < DETECTION_CONFIG['frame_delay']:
            time.sleep(DETECTION_CONFIG['frame_delay'] - elapsed_time)

except KeyboardInterrupt:
    logging.info("Exiting...")

except Exception as e:
    logging.error(f"Error occurred: {e}")

finally:
    cap.release()
    logging.info("Camera released. Program terminated.")
