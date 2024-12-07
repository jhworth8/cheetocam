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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("cat_detector.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting cat detection system...")

SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
PHONE_RECIPIENTS = [r.strip() for r in os.getenv('PHONE_RECIPIENTS', '').split(',') if r.strip()]
EMAIL_RECIPIENTS = [r.strip() for r in os.getenv('EMAIL_RECIPIENTS', '').split(',') if r.strip()]
IMGBB_API_KEY = os.getenv('IMGBB_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

ENABLE_GEMINI = int(os.getenv('ENABLE_GEMINI', '1'))
ENABLE_EMAIL_RESPONSE = int(os.getenv('ENABLE_EMAIL_RESPONSE', '1'))
ENABLE_CAT_DETECTION = int(os.getenv('ENABLE_CAT_DETECTION', '1'))
ENABLE_EMAIL_CHECK = int(os.getenv('ENABLE_EMAIL_CHECK', '1'))
ENABLE_ALERT_SENDING = int(os.getenv('ENABLE_ALERT_SENDING', '1'))

DETECTION_DURATION = int(os.getenv('DETECTION_DURATION', '5'))
COOLDOWN_DURATION = int(os.getenv('COOLDOWN_DURATION', '30'))
FRAME_DELAY = float(os.getenv('FRAME_DELAY', '0.2'))
EMAIL_CHECK_INTERVAL = int(os.getenv('EMAIL_CHECK_INTERVAL', '5'))

yolo_dir = os.getenv('YOLO_DIR', 'yolo')
weights_path = os.path.join(yolo_dir, 'yolov3-tiny.weights')
config_path = os.path.join(yolo_dir, 'yolov3-tiny.cfg')
names_path = os.path.join(yolo_dir, 'coco.names')

IMAP_SERVER = 'imap.gmail.com'
IMAP_PORT = 993

if ENABLE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)

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
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content([prompt, sample_file])
        logging.info("Gemini response received.")
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Could not generate description."

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
                        # Process with Gemini if enabled
                        gemini_response = ""
                        if ENABLE_GEMINI:
                            prompt = "This is an image captured in response to your email inquiry. Describe what you see in short sentences."
                            gemini_response = get_gemini_response(image_path, prompt)

                        eastern = timezone('US/Eastern')
                        subject = "Here's what's going on!"
                        message = 'Current situation outside the door...'
                        if gemini_response:
                            message += f"\n\nGemini Response:\n{gemini_response}"
                        send_email_with_attachments(
                            image_paths=[image_path],
                            subject=subject,
                            message=message,
                            phone_recipients=[],
                            email_recipients=[sender]
                        )
                mail.store(email_id, '+FLAGS', '\\Seen')
        mail.logout()
    except Exception as e:
        logging.error(f"Email check error: {e}")

# Initialize camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows
cap = cv2.VideoCapture("/dev/video0") # For Raspberry Pi
if not cap.isOpened():
    logging.error("Cannot open camera")
    exit()

# Variables for toggling detection states
last_cat_state = False
toggle_count = 0
cooldown_end_time = 0.0
last_email_check_time = 0.0

logging.info("Camera initialized. Beginning main loop...")

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

            # Log the detection state
            if cat_detected_now:
                logging.info("Cat detected in current frame.")

            # Check if state changed
            if cat_detected_now != last_cat_state:
                toggle_count += 1
                # Determine descriptive states
                state_from = "Detected" if last_cat_state else "Not Detected"
                state_to = "Detected" if cat_detected_now else "Not Detected"
                
                # Log the state change with descriptive terms
                logging.info(f"Cat detection state changed from {state_from} to {state_to}. Toggle count: {toggle_count}")
                
                last_cat_state = cat_detected_now


            # Once toggles reach 3, consider it a final detection event
            if toggle_count >= 3:
                logging.info("Cat detection confirmed after 3 toggles.")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_image_path = f'cat_detected_{timestamp}.jpg'
                cv2.imwrite(full_image_path, frame)
                image_paths = [full_image_path]

                cropped_image_paths = []
                for idx, i in enumerate(indexes.flatten()):
                    x, y, w, h = boxes[i]
                    cropped = frame[y:y+h, x:x+w]
                    cropped_image_path = f'cat_cropped_{timestamp}_{idx}.jpg'
                    cv2.imwrite(cropped_image_path, cropped)
                    image_paths.append(cropped_image_path)
                    cropped_image_paths.append(cropped_image_path)

                gemini_response = ""
                if ENABLE_GEMINI and cropped_image_paths:
                    prompt = "This is an image of a cat detection camera! What do you see? Short sentences."
                    gemini_response = get_gemini_response(cropped_image_paths[0], prompt)

                eastern = timezone('US/Eastern')
                subject = f"Cat Detected at {datetime.now(eastern).strftime('%I:%M %p ET')}!"
                message = "A cat has been detected outside your door!"
                if gemini_response:
                    message += f"\n\nGemini Response:\n{gemini_response}"

                send_email_with_attachments(
                    image_paths=image_paths,
                    subject=subject,
                    message=message,
                    phone_recipients=PHONE_RECIPIENTS,
                    email_recipients=EMAIL_RECIPIENTS
                )

                cooldown_end_time = current_time + COOLDOWN_DURATION
                toggle_count = 0  # Reset toggle count after final detection

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
