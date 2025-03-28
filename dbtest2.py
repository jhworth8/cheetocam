#!/usr/bin/env python3
import firebase_admin
from firebase_admin import credentials, db
import time
from datetime import datetime
import base64
from PIL import Image
import io

# Firebase setup: update with your Firebase Realtime Database URL.
FIREBASE_CONFIG = {
    "databaseURL": "https://cat-detector-77f57-default-rtdb.firebaseio.com/"
}

# Initialize Firebase Admin SDK with your service account key.
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, FIREBASE_CONFIG)

def generate_blank_image(width=512, height=512, color="white"):
    """Creates a blank image of the specified size and color."""
    img = Image.new("RGB", (width, height), color)
    return img

def image_to_base64(img):
    """Converts a PIL Image to a Base64-encoded string in PNG format."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def upload_blank_detection():
    """Generates a blank detection record and uploads it to Firebase."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch = int(time.time())
    blank_image = generate_blank_image()
    image_base64 = image_to_base64(blank_image)

    # Prepare the detection record.
    detection_data = {
        "timestamp": timestamp,
        "epoch": epoch,
        "main_image": image_base64,
        "detectionTemp": None,
        "detectionWeather": None,
        "detectionIcon": None,
        "gemini_response": "Blank detection record."
    }

    # Push the detection record to the "detections" node.
    ref = db.reference("detections")
    new_ref = ref.push(detection_data)
    print(f"Blank detection record uploaded with key: {new_ref.key}")

if __name__ == "__main__":
    upload_blank_detection()
