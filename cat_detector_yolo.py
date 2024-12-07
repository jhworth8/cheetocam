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

# Configuration from environment
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
PHONE_RECIPIENTS = [r.strip() for r in os.getenv('PHONE_RECIPIENTS', '').split(',') if r.strip()]
EMAIL_RECIPIENTS = [r.strip() for r in os.getenv('EMAIL_RECIPIENTS', '').split(',') if r.strip()]
IMGBB_API_KEY = os.getenv('IMGBB_API_KEY')
YOLO_DIR = os.getenv('YOLO_DIR', 'yolo')

DETECTION_DURATION = float(os.getenv('DETECTION_DURATION', '3'))
ALERT_DURATION = float(os.getenv('ALERT_DURATION', '120'))
FRAME_DELAY = float(os.getenv('FRAME_DELAY', '0.2'))
EMAIL_CHECK_INTERVAL = float(os.getenv('EMAIL_CHECK_INTERVAL', '5'))

IMAP_SERVER = 'imap.gmail.com'
IMAP_PORT = 993

weights_path = os.path.join(YOLO_DIR, 'yolov3-tiny.weights')
config_path = os.path.join(YOLO_DIR, 'yolov3-tiny.cfg')
names_path = os.path.join(YOLO_DIR, 'coco.names')

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

def upload_image_to_imgbb(image_path, api_key):
    try:
        with open(image_path, "rb") as file:
            encoded_image = base64.b64encode(file.read()).decode('utf-8')

        payload = {
            "key": api_key,
            "image": encoded_image,
        }

        response = requests.post("https://api.imgbb.com/1/upload", data=payload, timeout=30)

        if response.status_code == 200:
            json_response = response.json()
            image_url = json_response['data']['url']
            logging.info(f"Image uploaded to imgbb: {image_url}")
            return image_url
        else:
            logging.error(f"Failed to upload image to imgbb. Status Code: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return None
    except requests.exceptions.Timeout:
        logging.error("Image upload to imgbb timed out.")
        return None
    except Exception as e:
        logging.error(f"Exception during image upload to imgbb: {e}")
        return None

def send_email_with_attachments(image_paths, subject, message, phone_recipients, email_recipients):
    all_recipients = phone_recipients + email_recipients

    for recipient in all_recipients:
        msg = EmailMessage()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.set_content(message)

        for image_path in image_paths:
            if not os.path.exists(image_path):
                logging.warning(f"Image path {image_path} does not exist. Skipping attachment.")
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
                logging.info(f"Email (MMS) sent to {recipient}! Subject: {subject}")
        except Exception as e:
            logging.error(f"Failed to send email (MMS) to {recipient}: {e}")

def check_email(cap):
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
                    logging.error(f"Failed to fetch email ID {email_id}")
                    continue

                msg = email.message_from_bytes(msg_data[0][1])
                sender = email.utils.parseaddr(msg['From'])[1]
                logging.info(f"New email detected from: {sender}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f'email_triggered_{timestamp}.jpg'
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(image_path, frame)
                    logging.info(f"Captured image saved: {image_path}")
                else:
                    logging.error("Failed to capture image.")
                    continue

                eastern = timezone('US/Eastern')
                subject = "Here is what's going on!"
                message = 'Currently outside the door...'
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
        logging.error(f"Error in checking email: {e}")

cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    logging.error("Cannot open camera")
    exit()

state = 'waiting'
cat_detected_start_time = None
last_email_check_time = 0

try:
    while True:
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame.")
            break

        current_time = time.time()
        if current_time - last_email_check_time >= EMAIL_CHECK_INTERVAL:
            check_email(cap)
            last_email_check_time = current_time

        height, width, channels = frame.shape
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
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
        cat_detected = len(indexes) > 0

        if state == 'waiting':
            if cat_detected:
                if cat_detected_start_time is None:
                    cat_detected_start_time = current_time
                    logging.info("Cat detected. Starting 5-second timer.")
                else:
                    elapsed = current_time - cat_detected_start_time
                    if elapsed >= DETECTION_DURATION:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        full_image_path = f'cat_detected_{timestamp}.jpg'
                        cv2.imwrite(full_image_path, frame)
                        logging.info("5 seconds of continuous detection. Taking full-frame picture.")

                        image_paths = [full_image_path]
                        for idx, i in enumerate(indexes.flatten()):
                            x, y, w, h = boxes[i]
                            cropped_image = frame[y:y+h, x:x+w]
                            cropped_image_path = f'cat_cropped_{timestamp}_{idx}.jpg'
                            cv2.imwrite(cropped_image_path, cropped_image)
                            image_paths.append(cropped_image_path)
                            logging.info(f"Cropped image saved: {cropped_image_path}")

                        eastern = timezone('US/Eastern')
                        subject = f"Cat Detected at {datetime.now(eastern).strftime('%I:%M %p ET')}!"
                        message = 'A cat has been detected outside your door!'
                        send_email_with_attachments(
                            image_paths=image_paths,
                            subject=subject,
                            message=message,
                            phone_recipients=PHONE_RECIPIENTS,
                            email_recipients=EMAIL_RECIPIENTS
                        )

                        state = 'waiting'
                        cat_detected_start_time = None
            else:
                if cat_detected_start_time is not None:
                    logging.info("Cat no longer detected. Resetting timer.")
                cat_detected_start_time = None

        elapsed_time = time.time() - loop_start_time
        if elapsed_time < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed_time)

except KeyboardInterrupt:
    logging.info("Quitting...")

except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    cap.release()
