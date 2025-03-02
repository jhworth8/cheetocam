import firebase_admin
from firebase_admin import credentials, db
import base64
import random
import time
from datetime import datetime
from PIL import Image, ImageDraw
import io

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")  # Replace with your actual JSON file
firebase_admin.initialize_app(cred, {"databaseURL": "https://cat-detector-77f57-default-rtdb.firebaseio.com/"})

# Reference to the "detections" node
detections_ref = db.reference("detections")

def generate_random_image():
    """Generates a random cat-like image as a placeholder."""
    img = Image.new("RGB", (200, 200), (255, 204, 102))  # Light orange "Cheeto" color
    draw = ImageDraw.Draw(img)
    
    # Draw random features (simulating a cat face)
    draw.ellipse((50, 50, 150, 150), fill=(0, 0, 0))  # Eyes
    draw.ellipse((80, 80, 120, 120), fill=(255, 255, 255))  # Pupils
    draw.rectangle((90, 140, 110, 160), fill=(0, 0, 0))  # Nose

    # Convert to base64
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format="JPEG")
    return base64.b64encode(img_byte_array.getvalue()).decode("utf-8")

def upload_detection_event():
    """Uploads a fake cat detection event to Firebase."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main_image = generate_random_image()

    # Simulating a response from Gemini AI
    gemini_response = random.choice([
        "This appears to be a domestic feline.",
        "The detected object resembles a cat!",
        "A fluffy creature has been detected!"
    ])

    # Simulating cropped images
    cropped_images = [{"image": generate_random_image()} for _ in range(random.randint(1, 3))]

    # Detection event data
    detection_event = {
        "timestamp": timestamp,
        "main_image": main_image,
        "gemini_response": gemini_response,
        "cropped_images": cropped_images
    }

    # Upload to Firebase
    detections_ref.push(detection_event)
    print(f"Uploaded new detection at {timestamp}")

if __name__ == "__main__":
    while True:
        upload_detection_event()
        time.sleep(10)  # Upload a new detection every 10 seconds
