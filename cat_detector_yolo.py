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
import requests  # Ensure you have the requests library installed
import base64
import logging

# ==============================
# Logging Configuration
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("cat_detector.log"),
        logging.StreamHandler()
    ]
)

# ==============================
# Configuration Section
# ==============================

# Email configuration
SENDER_EMAIL = 'acatwasdetected@gmail.com'
SENDER_PASSWORD = 'bnxh uwio rvhi mevk'  # **Use an App Password if using Gmail**

# Recipient configurations
PHONE_RECIPIENTS = [
    '+19012672008@tmomail.net',  # T-Mobile MMS gateway
    '2486068897@vtext.com',
    '19018489759@mms.att.net'    # AT&T MMS gateway (commented out)
]

EMAIL_RECIPIENTS = [
    'jhworth8@gmail.com',        # Additional email recipient
    'cardosie4@gmail.com'      # Additional email recipient (commented out)
]

# imgbb configuration
IMGBB_API_KEY = 'e39a6cc3627f4056f727e5bc47b0e051'  # Replace with your actual imgbb API key

# Paths to YOLO files
yolo_dir = 'yolo'  # Directory where YOLO files are stored
weights_path = os.path.join(yolo_dir, 'yolov3-tiny.weights')
config_path = os.path.join(yolo_dir, 'yolov3-tiny.cfg')
names_path = os.path.join(yolo_dir, 'coco.names')

# Detection timing configurations
DETECTION_DURATION = 3     # Seconds of continuous detection before alert
ALERT_DURATION = 120        # Seconds to wait before sending second alert (Unused now)
FRAME_DELAY = 0.2           # Delay between frames for ~5 FPS

# IMAP configuration
IMAP_SERVER = 'imap.gmail.com'
IMAP_PORT = 993
EMAIL_CHECK_INTERVAL = 5  # Seconds between email checks (reduced for more frequent checks)

# ==============================
# Load YOLO Model
# ==============================

# Load class names
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Ensure 'cat' is in the classes
if 'cat' not in classes:
    logging.error("'cat' class not found in COCO names.")
    raise ValueError("'cat' class not found in COCO names.")

# Get the index of 'cat' class
cat_class_id = classes.index('cat')

# Load YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU; for GPU, adjust accordingly

# Get output layer names
layer_names = net.getLayerNames()
# getUnconnectedOutLayers() returns 1-based indices
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ==============================
# Define Helper Functions
# ==============================

def upload_image_to_imgbb(image_path, api_key):
    """
    Uploads an image to imgbb.com and returns the image URL.

    :param image_path: Path to the image file to upload.
    :param api_key: Your imgbb API key.
    :return: Public URL of the uploaded image or None if upload fails.
    """
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
    """
    Sends emails with multiple image attachments to phone and email recipients.

    :param image_paths: List of image file paths to attach.
    :param subject: Subject of the email.
    :param message: Body content of the email.
    :param phone_recipients: List of phone recipient email addresses (Email-to-MMS gateways).
    :param email_recipients: List of regular email recipient addresses.
    """
    # Combine all recipients
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
                # Determine image subtype based on file extension
                _, ext = os.path.splitext(image_path)
                ext = ext.lower().replace('.', '')
                if ext not in ['jpg', 'jpeg', 'png']:
                    ext = 'jpeg'  # Default to jpeg if unknown
                msg.add_attachment(img_data, maintype='image', subtype=ext, filename=os.path.basename(image_path))

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
                logging.info(f"Email (MMS) sent to {recipient}! Subject: {subject}")
        except Exception as e:
            logging.error(f"Failed to send email (MMS) to {recipient}: {e}")

def check_email(cap):
    """
    Connects to the email inbox and checks for new emails.
    For each new email, capture an image and send it back to the sender.

    :param cap: OpenCV VideoCapture object.
    """
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(SENDER_EMAIL, SENDER_PASSWORD)
        mail.select('inbox')

        # Search for all unseen emails
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

                # Take a picture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f'email_triggered_{timestamp}.jpg'
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(image_path, frame)
                    logging.info(f"Captured image saved: {image_path}")
                else:
                    logging.error("Failed to capture image.")
                    continue

                # Send the captured image back to the sender
                eastern = timezone('US/Eastern')
                subject = "Here is what's going on!"  # Updated subject line
                message = 'Currently outside the door...'
                send_email_with_attachments(
                    image_paths=[image_path],
                    subject=subject,
                    message=message,
                    phone_recipients=[],  # No phone recipients in this response
                    email_recipients=[sender]
                )

                # Mark the email as seen
                mail.store(email_id, '+FLAGS', '\\Seen')

        mail.logout()
    except Exception as e:
        logging.error(f"Error in checking email: {e}")

# ==============================
# Initialize Camera
# ==============================

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend; adjust as needed
cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    logging.error("Cannot open camera")
    exit()

# ==============================
# Detection State Variables
# ==============================

state = 'waiting'  # Possible states: 'waiting'
cat_detected_start_time = None
last_email_check_time = 0  # Track the last time emails were checked

# ==============================
# Main Detection Loop
# ==============================

try:
    while True:
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame.")
            break

        # Check if it's time to check emails
        current_time = time.time()
        if current_time - last_email_check_time >= EMAIL_CHECK_INTERVAL:
            check_email(cap)
            last_email_check_time = current_time

        # Prepare the frame for YOLO
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Initialize lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Process YOLO detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == cat_class_id and confidence > 0.5:  # Threshold can be adjusted
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Ensure the bounding box is within the frame
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Max Suppression to eliminate redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Determine if a cat is detected in this frame
        cat_detected = len(indexes) > 0

        if state == 'waiting':
            if cat_detected:
                if cat_detected_start_time is None:
                    cat_detected_start_time = current_time
                    logging.info("Cat detected. Starting 5-second timer.")
                else:
                    elapsed = current_time - cat_detected_start_time
                    if elapsed >= DETECTION_DURATION:
                        # Take full-frame picture
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        full_image_path = f'cat_detected_{timestamp}.jpg'
                        cv2.imwrite(full_image_path, frame)
                        logging.info("5 seconds of continuous detection. Taking full-frame picture.")

                        # Initialize list of image paths to send
                        image_paths = [full_image_path]

                        # Save cropped images for each detected cat
                        for idx, i in enumerate(indexes.flatten()):
                            x, y, w, h = boxes[i]
                            cropped_image = frame[y:y+h, x:x+w]
                            cropped_image_path = f'cat_cropped_{timestamp}_{idx}.jpg'
                            cv2.imwrite(cropped_image_path, cropped_image)
                            image_paths.append(cropped_image_path)
                            logging.info(f"Cropped image saved: {cropped_image_path}")

                        # Send email (MMS) to phone recipients and email notifications
                        eastern = timezone('US/Eastern')
                        subject = f"Cat Detected at {datetime.now(eastern).strftime('%I:%M %p ET')}!"
                        message = 'A cat has been detected outside your door!'
                        send_email_with_attachments(
                            image_paths=image_paths,
                            subject=subject,  # This is for detection alerts
                            message=message,
                            phone_recipients=PHONE_RECIPIENTS,
                            email_recipients=EMAIL_RECIPIENTS
                        )

                        # Update state
                        state = 'waiting'  # Remain in 'waiting' to allow future detections
                        cat_detected_start_time = None  # Reset for next detection
            else:
                if cat_detected_start_time is not None:
                    logging.info("Cat no longer detected. Resetting timer.")
                cat_detected_start_time = None  # Reset timer if cat not detected

        # Optionally, log frame processing or detections here

        # Maintain ~5 FPS by adjusting frame_delay
        elapsed_time = time.time() - loop_start_time
        if elapsed_time < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed_time)

except KeyboardInterrupt:
    logging.info("Quitting...")

except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    cap.release()
    # No need to destroy windows since we're not displaying any
