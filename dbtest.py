import firebase_admin
from firebase_admin import credentials, storage, db
import numpy as np
from PIL import Image, ImageDraw
import random
import io
import time

# Firebase setup
FIREBASE_CONFIG = {
    "databaseURL": "https://cat-detector-77f57-default-rtdb.firebaseio.com/",
    "storageBucket": "cat-detector-77f57.firebasestorage.app"
}

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, FIREBASE_CONFIG)
bucket = storage.bucket()

def generate_squiggle_image():
    """Creates a 512x512 image with random squiggles."""
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for _ in range(20):  # Number of squiggles
        points = [(random.randint(0, 512), random.randint(0, 512)) for _ in range(5)]
        color = tuple(np.random.randint(0, 256, 3))  # Random RGB color
        draw.line(points, fill=color, width=5)

    return img

def upload_to_firebase(img):
    """Uploads the image to Firebase Storage and stores the URL in Realtime Database."""
    timestamp = str(int(time.time()))
    filename = f"squiggle_{timestamp}.png"

    # Convert image to bytes
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)

    # Upload to Firebase Storage
    blob = bucket.blob(f"squiggles/{filename}")
    blob.upload_from_file(img_byte_array, content_type='image/png')

    # Get public URL
    blob.make_public()
    img_url = blob.public_url

    # Store URL in Firebase Realtime Database
    ref = db.reference("squiggles")
    ref.push({"timestamp": timestamp, "url": img_url})

    print(f"Image uploaded: {img_url}")
    return img_url

if __name__ == "__main__":
    img = generate_squiggle_image()
    upload_to_firebase(img)
