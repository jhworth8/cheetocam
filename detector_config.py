#!/usr/bin/env python3
"""
Configuration file for the enhanced cat detection system with YOLOv11
This file contains all configuration settings, avoiding the need to edit .env files
"""

# =============================================================================
# EMAIL CONFIGURATION
# =============================================================================
EMAIL_CONFIG = {
    'sender_email': 'your_email@gmail.com',
    'sender_password': 'your_app_password',
    'phone_recipients': ['phone1@carrier.com', 'phone2@carrier.com'],
    'email_recipients': ['email1@gmail.com', 'email2@gmail.com'],
}

# =============================================================================
# API KEYS AND CREDENTIALS
# =============================================================================
API_CONFIG = {
    'gemini_api_key': 'your_gemini_api_key',
    'openweather_api_key': 'your_openweather_api_key',
    'supabase_url': 'your_supabase_url',
    'supabase_anon_key': 'your_supabase_anon_key',
    
    # Pushover credentials (hardcoded as requested)
    'pushover_user_key': 'ubr5phjr9wymwabf8sg1anmsj9a12k',
    'pushover_api_token': 'aqgxkbzacmm4mi8fpu49duzwnorfjf',
}

# =============================================================================
# DETECTION CONFIGURATION
# =============================================================================
DETECTION_CONFIG = {
    # Model selection
    'model_name': 'yolo11n.pt',  # YOLOv11 Nano for Raspberry Pi 5
    'fallback_model': 'yolov8n.pt',  # Fallback if YOLOv11 fails
    
    # Detection settings
    'detection_confidence': 0.5,
    'detection_classes': [
        # Animals
        'cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        
        # People
        'person',
        
        # Vehicles
        'car', 'truck', 'bus', 'train', 'bicycle', 'motorbike', 'boat', 'aeroplane',
        
        # Objects
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
        
        # Sports equipment
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        
        # Traffic and signs
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        
        # Bags and luggage
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    ],
    
    # Enable/disable features
    'enable_gemini': True,
    'enable_email_response': True,
    'enable_cat_detection': True,
    'enable_email_check': True,
    'enable_alert_sending': True,
    'enable_supabase_upload': True,
    'enable_multi_class_detection': True,
    
    # Timing settings
    'cooldown_duration': 30,  # seconds between detections
    'frame_delay': 0.2,  # seconds between frame processing
    'email_check_interval': 5,  # seconds between email checks
}

# =============================================================================
# WEATHER CONFIGURATION
# =============================================================================
WEATHER_CONFIG = {
    'latitude': 42.5467,
    'longitude': -83.2113,
    'timezone': 'US/Eastern',
}

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================
CAMERA_CONFIG = {
    'device': '/dev/video0',
    'resolution': (640, 480),  # Width, Height
    'fps': 30,
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s [%(levelname)s] %(message)s',
    'file': 'cat_detector.log',
    'console': True,
}

# =============================================================================
# PERFORMANCE OPTIMIZATION FOR RASPBERRY PI 5
# =============================================================================
PERFORMANCE_CONFIG = {
    'max_detection_classes': 20,  # Limit concurrent class detection for performance
    'image_resize': (416, 416),  # Resize input images for faster processing
    'batch_size': 1,  # Process one frame at a time
    'enable_gpu': False,  # Disable GPU acceleration on Pi 5
    'memory_limit_mb': 512,  # Limit memory usage
}

# =============================================================================
# NOTIFICATION CONFIGURATION
# =============================================================================
NOTIFICATION_CONFIG = {
    'enable_pushover': True,
    'enable_email': True,
    'enable_supabase': True,
    'priority_classes': ['cat', 'dog', 'person'],  # High priority classes
    'notification_cooldown': 60,  # seconds between notifications for same class
}

# =============================================================================
# BACKUP AND FALLBACK CONFIGURATION
# =============================================================================
BACKUP_CONFIG = {
    'yolo_dir': 'yolo',  # Directory for legacy YOLOv3 files
    'enable_fallback': True,  # Enable fallback to YOLOv3-tiny
    'backup_detection_classes': ['cat'],  # Classes to detect in fallback mode
}

# =============================================================================
# DEVELOPMENT AND DEBUGGING
# =============================================================================
DEBUG_CONFIG = {
    'enable_debug': False,
    'save_debug_images': False,
    'debug_image_dir': 'debug_images',
    'log_detection_details': False,
    'performance_monitoring': False,
}
