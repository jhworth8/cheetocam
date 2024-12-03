import cv2
import os
import smtplib
from email.message import EmailMessage
import time
from datetime import datetime
from pytz import timezone
import numpy as np

# Email configuration
SENDER_EMAIL = 'acatwasdetected@gmail.com'
SENDER_PASSWORD = 'bnxh uwio rvhi mevk'  # Ensure this is an App Password if using Gmail
RECIPIENT_EMAIL = 'jhworth8@gmail.com'

# Paths to YOLO files
yolo_dir = 'yolo'  # Directory where YOLO files are stored
weights_path = os.path.join(yolo_dir, 'yolov3.weights')
config_path = os.path.join(yolo_dir, 'yolov3.cfg')
names_path = os.path.join(yolo_dir, 'coco.names')

# Load class names
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Ensure 'cat' is in the classes
if 'cat' not in classes:
    raise ValueError("'cat' class not found in COCO names.")

# Get the index of 'cat' class
cat_class_id = classes.index('cat')

# Load YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU; for GPU, adjust accordingly

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

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

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            print(f"Email sent! Subject: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Initialize camera
cap = cv2.VideoCapture(0)

# Variables for cooldown and frame throttling
# cooldown_time = 30  # Cooldown time in seconds  # Commented out # Modified
# last_detection_time = 0  # Commented out # Modified
frame_delay = 0.02  # Reduced delay for higher FPS # Modified from 0.5 to 0.02 (~50 FPS)

try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Check if enough time has passed since the last detection
        # if time.time() - last_detection_time > cooldown_time:  # Commented out # Modified
        #     detection logic is now always active without cooldown
        height, width, channels = frame.shape

        # Create a blob and perform a forward pass
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Initialize lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

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

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Max Suppression to eliminate redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                # Draw bounding box around detected cat
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Optionally, add label and confidence
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
            # image_path = 'cat_detected.jpg'  # Optionally save image
            # cv2.imwrite(image_path, frame)
            # send_email_with_attachment(image_path)  # Commented out # Modified
            print("Cat detected!")
            # last_detection_time = time.time()  # Commented out # Modified

        # Show the current frame
        cv2.imshow('Cat Detection', frame)

        # Maintain higher FPS by reducing delay
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_delay:
            time.sleep(frame_delay - elapsed_time)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
