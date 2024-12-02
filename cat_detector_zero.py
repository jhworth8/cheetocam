import cv2
import os
import smtplib
from email.message import EmailMessage
import time
from datetime import datetime
from pytz import timezone

# Email configuration
SENDER_EMAIL = 'acatwasdetected@gmail.com'
SENDER_PASSWORD = 'bnxh uwio rvhi mevk'
RECIPIENT_EMAILS = ['jhworth8@gmail.com', 'cardosie4@gmail.com']

# Load the Haar Cascade for cat face detection
cascade_path = 'haarcascade_frontalcatface.xml'
cat_cascade = cv2.CascadeClassifier(cascade_path)

# Function to send email with attachment
def send_email_with_attachment(image_path):
    # Get the current Eastern Time
    eastern = timezone('US/Eastern')
    current_time = datetime.now(eastern).strftime("%I:%M %p ET")
    subject = f"Cat Detected at {current_time}!"

    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.set_content('A cat has been detected outside your door. See the attached image.')

    with open(image_path, 'rb') as img:
        img_data = img.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        print(f"Email sent! Subject: {subject}")

# Initialize camera
cap = cv2.VideoCapture(0)

# Variables for cooldown and frame throttling
cooldown_time = 30  # Cooldown time in seconds
last_detection_time = 0
frame_delay = 0.5  # Delay between frames for ~2 FPS

try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Check if enough time has passed since the last detection
        if time.time() - last_detection_time > cooldown_time:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

            if len(cats) > 0:  # If a cat is detected
                for (x, y, w, h) in cats:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                image_path = 'cat_detected.jpg'
                cv2.imwrite(image_path, frame)
                send_email_with_attachment(image_path)
                print("Cat detected!")
                last_detection_time = time.time()  # Reset the cooldown timer

        # Maintain 2 FPS by adding a delay
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_delay:
            time.sleep(frame_delay - elapsed_time)

        # Quit loop on Ctrl+C (since no GUI to quit manually)
except KeyboardInterrupt:
    print("Program terminated by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
