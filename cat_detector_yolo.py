import cv2
import os
import smtplib
from email.message import EmailMessage
import time
from datetime import datetime
from pytz import timezone
import numpy as np
import requests
import logging
import random
from typing import Dict, Optional
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
YOLO_DIR = os.getenv('YOLO_DIR', 'yolo')

DETECTION_DURATION = float(os.getenv('DETECTION_DURATION', '3'))
ALERT_DURATION = float(os.getenv('ALERT_DURATION', '120'))
FRAME_DELAY = float(os.getenv('FRAME_DELAY', '0.2'))
TIMEZONE = os.getenv('TIMEZONE', 'US/Eastern')
WEATHER_LATITUDE = os.getenv('WEATHER_LATITUDE')
WEATHER_LONGITUDE = os.getenv('WEATHER_LONGITUDE')

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

WEATHER_CODE_MAP: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Light snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with light hail",
    99: "Thunderstorm with heavy hail",
}

WEATHER_EMOJI_MAP: Dict[int, str] = {
    0: "☀️",
    1: "🌤️",
    2: "⛅",
    3: "☁️",
    45: "🌫️",
    48: "🌁",
    51: "🌦️",
    53: "🌦️",
    55: "🌧️",
    56: "🌧️",
    57: "🌧️",
    61: "🌧️",
    63: "🌧️",
    65: "🌧️",
    66: "🌧️",
    67: "🌧️",
    71: "❄️",
    73: "❄️",
    75: "❄️",
    77: "🌨️",
    80: "🌧️",
    81: "🌧️",
    82: "🌩️",
    85: "❄️",
    86: "❄️",
    95: "⛈️",
    96: "⛈️",
    99: "⛈️",
}

MOOD_BOOSTERS = [
    "Fun Fact: Cats have fewer taste buds than dogs, but they're better at spotting the laser pointer!",
    "Joke: Why don't cats play poker in the jungle? Too many cheetahs!",
    "Trivia: A group of kittens is called a kindle—your daily vocabulary boost!",
    "Inspiration: 'In ancient times cats were worshipped as gods; they have not forgotten this.' — Terry Pratchett",
    "Playful Prompt: Take a dance break! Bonus points if your cat joins in.",
    "Self-Care Spark: Sip some water and stretch—future you will say me-wow!",
]


def interpret_weather(code: int) -> str:
    return WEATHER_CODE_MAP.get(code, "Mystery weather")


def weather_emoji(code: int) -> str:
    return WEATHER_EMOJI_MAP.get(code, "🌈")


def fetch_weather_snapshot() -> Optional[Dict[str, str]]:
    if not WEATHER_LATITUDE or not WEATHER_LONGITUDE:
        logging.info("Weather coordinates not provided; skipping weather lookup.")
        return None

    params = {
        "latitude": WEATHER_LATITUDE,
        "longitude": WEATHER_LONGITUDE,
        "current_weather": "true",
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": TIMEZONE,
    }

    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current_weather")
        daily = payload.get("daily", {})
        if not current:
            logging.warning("Weather lookup succeeded but missing current weather data.")
            return None

        code = int(current.get("weathercode", -1))
        description = interpret_weather(code)
        emoji = weather_emoji(code)
        temperature = current.get("temperature")
        wind_speed = current.get("windspeed")
        high = None
        low = None

        temps_max = daily.get("temperature_2m_max")
        temps_min = daily.get("temperature_2m_min")
        if isinstance(temps_max, list) and temps_max:
            high = temps_max[0]
        if isinstance(temps_min, list) and temps_min:
            low = temps_min[0]

        snapshot = {
            "description": description,
            "emoji": emoji,
            "temperature": f"{temperature:.0f}°C" if isinstance(temperature, (int, float)) else "N/A",
            "wind": f"{wind_speed:.0f} km/h" if isinstance(wind_speed, (int, float)) else "N/A",
        }

        if high is not None and low is not None:
            snapshot["range"] = f"{low:.0f}°C → {high:.0f}°C"

        return snapshot
    except requests.RequestException as exc:
        logging.error(f"Unable to fetch weather data: {exc}")
    except ValueError as exc:
        logging.error(f"Weather data parsing error: {exc}")

    return None


def choose_mood_booster() -> str:
    booster = random.choice(MOOD_BOOSTERS)
    logging.debug(f"Selected mood booster: {booster}")
    return booster


def themed_greeting(event_time: datetime) -> str:
    hour = event_time.hour
    if 5 <= hour < 12:
        vibe = "Morning Marvel"
    elif 12 <= hour < 17:
        vibe = "Afternoon Adventure"
    elif 17 <= hour < 21:
        vibe = "Twilight Triumph"
    else:
        vibe = "Moonlit Magic"
    return f"{vibe} Alert!"


def current_event_time() -> datetime:
    try:
        tz = timezone(TIMEZONE)
    except Exception:
        logging.warning(f"Invalid timezone '{TIMEZONE}' provided. Falling back to UTC.")
        tz = timezone('UTC')
    return datetime.now(tz)


def build_notification_content(event_time: datetime, weather: Optional[Dict[str, str]]) -> Dict[str, str]:
    booster = choose_mood_booster()
    friendly_timestamp = event_time.strftime("%A, %B %d • %I:%M %p %Z")
    header = themed_greeting(event_time)

    weather_lines = []
    if weather:
        weather_lines.append(f"Weather: {weather['emoji']}  {weather['description']} at {weather['temperature']}")
        if "range" in weather:
            weather_lines.append(f"Day Range: {weather['range']}")
        weather_lines.append(f"Wind: {weather['wind']}")
    else:
        weather_lines.append("Weather: 🌈  Unable to load live weather—enjoy the mystery!")

    narrative = [
        "┏━━━━━━━━━━━━━━━━━━━━━━━┓",
        "┃  🐾 Cat Cam Spotlight  ┃",
        "┗━━━━━━━━━━━━━━━━━━━━━━━┛",
        f"Timestamp: {friendly_timestamp}",
        *weather_lines,
        "",
        "Sighting Summary:",
        "Your door-side paparazzi caught a feline friend striking a pose!",
        "",
        f"Mood Booster: {booster}",
        "",
        "Pro Tip: Share this moment with the household scoreboard for bonus bragging rights!",
    ]

    subject_weather = weather['emoji'] if weather else '🐱'
    subject = f"{subject_weather} {header} — Cat Detected!"

    return {
        "subject": subject,
        "message": "\n".join(narrative),
    }

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

cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    logging.error("Cannot open camera")
    exit()

state = 'waiting'
cat_detected_start_time = None

try:
    while True:
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame.")
            break

        current_time = time.time()

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

                        event_time = current_event_time()
                        weather_snapshot = fetch_weather_snapshot()
                        notification = build_notification_content(event_time, weather_snapshot)

                        send_email_with_attachments(
                            image_paths=image_paths,
                            subject=notification['subject'],
                            message=notification['message'],
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
