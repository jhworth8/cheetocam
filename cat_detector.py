import cv2
import numpy as np
import tensorflow as tf
import os
import smtplib
from email.message import EmailMessage

# Email configuration
SENDER_EMAIL = 'acatwasdetected@gmail.com'
SENDER_PASSWORD = 'CheetoTheCat123!'
RECIPIENT_EMAIL = 'jhworth8@gmail.com'

# Load the pre-trained model
MODEL_PATH = 'cat_detector_updated.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess frame for model 
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Adjust based on model input size
    normalized_frame = resized_frame / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)
    return input_data

# Function to capture and save image
def capture_image(frame):
    image_path = 'cat_detected.jpg'
    cv2.imwrite(image_path, frame)
    return image_path

# Function to send email with attachment
def send_email_with_attachment(image_path):
    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = 'Cat Detected!'
    msg.set_content('A cat has been detected outside your door. See the attached image.')

    with open(image_path, 'rb') as img:
        img_data = img.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        print('Email sent!')

# Initialize camera
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_frame(frame)
        prediction = model.predict(input_data)

        if prediction[0][0] > 0.5:  # Adjust threshold if necessary
            image_path = capture_image(frame)
            send_email_with_attachment(image_path)
            print("Cat detected!")

        cv2.imshow('Cat Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
