import cv2
import os
import smtplib
from email.message import EmailMessage

# Email configuration
SENDER_EMAIL = 'acatwasdetected@gmail.com'
SENDER_PASSWORD = 'CheetoTheCat123!'
RECIPIENT_EMAIL = 'jhworth8@gmail.com'

# Load the Haar Cascade for cat face detection
cascade_path = 'haarcascade_frontalcatface.xml'
cat_cascade = cv2.CascadeClassifier(cascade_path)

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

        for (x, y, w, h) in cats:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            image_path = 'cat_detected.jpg'
            cv2.imwrite(image_path, frame)
            send_email_with_attachment(image_path)
            print("Cat detected!")

        cv2.imwrite('current_frame.jpg', frame)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
