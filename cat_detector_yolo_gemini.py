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
from supabase import create_client, Client  # Import for Supabase

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

logging.info("Starting cat detection system with Supabase integration...")

# Email and API configuration
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
PHONE_RECIPIENTS = [r.strip() for r in os.getenv('PHONE_RECIPIENTS', '').split(',') if r.strip()]
EMAIL_RECIPIENTS = [r.strip() for r in os.getenv('EMAIL_RECIPIENTS', '').split(',') if r.strip()]
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')  # Weather API key

ENABLE_GEMINI = int(os.getenv('ENABLE_GEMINI', '1'))
ENABLE_EMAIL_RESPONSE = int(os.getenv('ENABLE_EMAIL_RESPONSE', '1'))
ENABLE_CAT_DETECTION = int(os.getenv('ENABLE_CAT_DETECTION', '1'))
ENABLE_EMAIL_CHECK = int(os.getenv('ENABLE_EMAIL_CHECK', '1'))
ENABLE_ALERT_SENDING = int(os.getenv('ENABLE_ALERT_SENDING', '1'))
ENABLE_SUPABASE_UPLOAD = int(os.getenv('ENABLE_SUPABASE_UPLOAD', '1'))

COOLDOWN_DURATION = int(os.getenv('COOLDOWN_DURATION', '30'))
FRAME_DELAY = float(os.getenv('FRAME_DELAY', '0.2'))
EMAIL_CHECK_INTERVAL = int(os.getenv('EMAIL_CHECK_INTERVAL', '5'))

# YOLO configuration paths
yolo_dir = os.getenv('YOLO_DIR', 'yolo')
weights_path = os.path.join(yolo_dir, 'yolov3-tiny.weights')
config_path = os.path.join(yolo_dir, 'yolov3-tiny.cfg')
names_path = os.path.join(yolo_dir, 'coco.names')

IMAP_SERVER = 'imap.gmail.com'
IMAP_PORT = 993

# Initialize Gemini API if enabled
if ENABLE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Supabase client (using supabase-py)
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', '')
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Initialize YOLO for cat detection
if ENABLE_CAT_DETECTION:
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

def get_gemini_response(image_path, prompt):
    if not ENABLE_GEMINI:
        return ""
    try:
        sample_file = PIL.Image.open(image_path)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")
        response = model.generate_content([prompt, sample_file])
        logging.info("Gemini response received.")
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Could not generate description."

def upload_detection_to_supabase(timestamp, gemini_response, main_image_path, detectionTemp=None, detectionWeather=None, detectionIcon=None):
    if not ENABLE_SUPABASE_UPLOAD:
        return
    try:
        with open(main_image_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        epoch = time.time()
        # Use lowercase keys to match your schema: detectiontemp, detectionweather, detectionicon
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
                        # Fetch weather data for email-triggered detection
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
                        # Upload this detection to Supabase with weather data
                        upload_detection_to_supabase(timestamp, gemini_response, image_path, detectionTemp=temp, detectionWeather=weather, detectionIcon=icon)
                mail.store(email_id, '+FLAGS', '\\Seen')
        mail.logout()
    except Exception as e:
        logging.error(f"Email check error: {e}")

# Initialize camera (modify as needed for your platform)
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
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes = []
            confidences = []
            class_ids = []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == cat_class_id and confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = max(0, int(center_x - w / 2))
                        y = max(0, int(center_y - h / 2))
                        w = min(w, width - x)
                        h = min(h, height - y)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            cat_detected_now = (len(indexes) > 0)

            if cat_detected_now:
                logging.info("Cat detected by YOLO. Processing detection...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_image_path = f'cat_detected_{timestamp}.jpg'
                cv2.imwrite(full_image_path, frame)

                gemini_response = ""
                if ENABLE_GEMINI:
                    prompt = ("Please provide a clear and concise description of the scene captured. "
                              "Use short sentences to describe what you see.")
                    gemini_response = get_gemini_response(full_image_path, prompt)

                # Fetch weather data when a detection is captured
                temp, weather, icon = fetch_weather_data()

                if gemini_response and "cat" in gemini_response.lower():
                    logging.info("Gemini confirmed the presence of a cat. Uploading detection to Supabase...")
                    send_email_with_attachments(
                        image_paths=[full_image_path],
                        subject=f"Cat Detected at {datetime.now(timezone('US/Eastern')).strftime('%I:%M %p ET')}",
                        message="A cat has been detected outside your door!\n\n" + gemini_response + (f"\nWeather: {temp} °F, {weather}" if temp and weather else ""),
                        phone_recipients=PHONE_RECIPIENTS,
                        email_recipients=EMAIL_RECIPIENTS
                    )
                    upload_detection_to_supabase(timestamp, gemini_response, full_image_path, detectionTemp=temp, detectionWeather=weather, detectionIcon=icon)
                    cooldown_end_time = current_time + COOLDOWN_DURATION

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
