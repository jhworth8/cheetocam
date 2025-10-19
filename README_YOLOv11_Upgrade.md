# YOLOv11 Upgrade for Raspberry Pi 5

This upgrade replaces the old YOLOv3-tiny model with the modern YOLOv11n model, providing:

- **80 object classes** with comprehensive detection capabilities
- **Better detection accuracy** and speed than previous versions
- **Smaller model size** (~6MB vs ~33MB for YOLOv3-tiny)
- **Modern architecture** with improved performance
- **Multi-class detection** support with extensive class list
- **Configuration file** instead of .env for easier management

## What's New

### Enhanced Detection Capabilities
- **Primary focus**: Cats (maintained from original)
- **Comprehensive class list**: 80+ object classes including:
  - **Animals**: cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
  - **People**: person
  - **Vehicles**: car, truck, bus, train, bicycle, motorbike, boat, aeroplane
  - **Objects**: bottle, cup, chair, laptop, cell phone, book, clock, etc.
  - **Sports equipment**: frisbee, skis, snowboard, sports ball, kite, etc.
  - **Traffic signs**: traffic light, fire hydrant, stop sign, parking meter
  - **Food items**: banana, apple, sandwich, orange, pizza, donut, cake, etc.
- **Configurable detection**: Set `detection_classes` in `detector_config.py`
- **Better accuracy**: YOLOv11 provides superior detection performance

### Performance Improvements
- **Faster inference**: Optimized for Raspberry Pi 5
- **Lower memory usage**: More efficient model architecture
- **Better real-time performance**: Improved frame processing
- **Performance monitoring**: Built-in optimization features

### Configuration Management
- **No .env file needed**: All settings in `detector_config.py`
- **Easy configuration**: Edit one Python file instead of environment variables
- **Comprehensive settings**: All options documented and organized

## Installation

### 1. Install Dependencies
```bash
# Make the installation script executable
chmod +x install_yolov11.sh

# Run the installation script
./install_yolov11.sh
```

### 2. Configure Settings
```bash
# Edit the configuration file
nano detector_config.py
```

### 3. Key Configuration Options

#### Detection Settings
```python
DETECTION_CONFIG = {
    'model_name': 'yolo11n.pt',  # YOLOv11 Nano for Raspberry Pi 5
    'detection_confidence': 0.5,
    'detection_classes': [
        'cat', 'dog', 'person', 'bird', 'car', 'truck', 'bicycle', 
        'motorbike', 'bus', 'train', 'bottle', 'cup', 'chair', 
        'laptop', 'cell phone', 'book', 'clock', 'vase', 'scissors',
        # ... and many more
    ],
    'enable_multi_class_detection': True,
    'cooldown_duration': 30,
    'frame_delay': 0.2,
}
```

#### Email Configuration
```python
EMAIL_CONFIG = {
    'sender_email': 'your_email@gmail.com',
    'sender_password': 'your_app_password',
    'phone_recipients': ['phone1@carrier.com', 'phone2@carrier.com'],
    'email_recipients': ['email1@gmail.com', 'email2@gmail.com'],
}
```

#### API Configuration
```python
API_CONFIG = {
    'gemini_api_key': 'your_gemini_api_key',
    'openweather_api_key': 'your_openweather_api_key',
    'supabase_url': 'your_supabase_url',
    'supabase_anon_key': 'your_supabase_anon_key',
}
```

## Usage

### Basic Usage
```bash
# Run the upgraded detector
python3 cat_detector_yolo11.py
```

### Performance Monitoring
```bash
# Monitor system performance
python3 performance_monitor.py
```

### Run as Service
```bash
# Copy service file
sudo cp cat-detector.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable cat-detector.service
sudo systemctl start cat-detector.service

# Check status
sudo systemctl status cat-detector.service
```

## Performance Optimization

### For Raspberry Pi 5
1. **Use YOLOv11n**: The nano version is optimized for edge devices
2. **Adjust resolution**: Lower camera resolution for better performance
3. **Tune confidence**: Higher confidence = fewer false positives
4. **Frame delay**: Increase `frame_delay` if CPU usage is high
5. **Limit classes**: Use `max_detection_classes` to limit concurrent detection

### Recommended Settings for Pi 5
```python
PERFORMANCE_CONFIG = {
    'max_detection_classes': 20,  # Limit concurrent class detection
    'image_resize': (416, 416),  # Resize input images for faster processing
    'frame_delay': 0.3,  # Increase if CPU usage is high
    'enable_gpu': False,  # Disable GPU acceleration on Pi 5
    'memory_limit_mb': 512,  # Limit memory usage
}
```

## Troubleshooting

### Model Download Issues
If the YOLOv11n model fails to download:
```bash
# Manual download
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

### Performance Issues
1. **Run performance monitor**: `python3 performance_monitor.py`
2. **Check CPU temperature**: Should be < 70°C
3. **Monitor memory usage**: Should have > 100MB available
4. **Adjust settings**: Increase `frame_delay` or reduce `detection_classes`

### Fallback System
The system automatically falls back through multiple models:
1. **YOLOv11n** (primary)
2. **YOLOv8n** (fallback)
3. **YOLOv3-tiny** (final fallback)

## File Structure
```
cheetocam/
├── cat_detector_yolo11.py      # New YOLOv11 detector
├── detector_config.py          # Configuration file
├── cat_detector_yolo_v8.py     # YOLOv8 detector (backup)
├── cat_detector_yolo_gemini.py # Original detector (backup)
├── cat_detector_yolo.py        # Original detector (backup)
├── performance_monitor.py       # Performance monitoring
├── install_yolov11.sh          # Installation script
├── cat-detector.service        # Systemd service file
├── requirements.txt            # Updated dependencies
└── yolo/                       # Legacy YOLOv3 files (fallback)
    ├── yolov3-tiny.weights
    ├── yolov3-tiny.cfg
    └── coco.names
```

## Migration from Previous Versions

### Automatic Migration
The new system automatically:
1. Tries to load YOLOv11n first
2. Falls back to YOLOv8n if YOLOv11 fails
3. Falls back to YOLOv3-tiny if YOLOv8 fails
4. Maintains all existing functionality
5. Adds new multi-class detection capabilities

### Configuration Changes
- **No .env file**: All settings in `detector_config.py`
- **`detection_classes`**: Comprehensive list of 80+ classes
- **`detection_confidence`**: More precise confidence control
- **`enable_multi_class_detection`**: Enable/disable multi-class mode
- **Performance settings**: Built-in optimization options

## Support

### Logs
Check the log file for detailed information:
```bash
tail -f cat_detector.log
```

### Service Management
```bash
# View logs
sudo journalctl -u cat-detector.service -f

# Restart service
sudo systemctl restart cat-detector.service

# Stop service
sudo systemctl stop cat-detector.service
```

## Performance Comparison

| Model | Size | Classes | Speed (Pi 5) | Accuracy | Features |
|-------|------|---------|--------------|----------|----------|
| YOLOv3-tiny | ~33MB | 80 | ~3 FPS | Good | Basic |
| YOLOv8n | ~6MB | 80 | ~8-12 FPS | Better | Enhanced |
| YOLOv11n | ~6MB | 80 | ~10-15 FPS | Best | Latest |

The YOLOv11n model provides the best performance while using minimal memory and storage space.

## Configuration Examples

### Minimal Configuration (Cats Only)
```python
DETECTION_CONFIG = {
    'detection_classes': ['cat'],
    'enable_multi_class_detection': False,
    'detection_confidence': 0.6,
}
```

### Comprehensive Configuration (All Classes)
```python
DETECTION_CONFIG = {
    'detection_classes': [
        'cat', 'dog', 'person', 'bird', 'car', 'truck', 'bicycle', 'motorbike',
        'bus', 'train', 'boat', 'aeroplane', 'bottle', 'cup', 'chair', 'laptop',
        'cell phone', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'sofa', 'pottedplant',
        'bed', 'diningtable', 'toilet', 'tvmonitor', 'mouse', 'remote',
        'keyboard', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'hair drier', 'toothbrush', 'wine glass', 'fork', 'knife', 'spoon', 'bowl'
    ],
    'enable_multi_class_detection': True,
    'detection_confidence': 0.5,
}
```

### Performance-Optimized Configuration
```python
DETECTION_CONFIG = {
    'detection_classes': ['cat', 'dog', 'person', 'bird', 'car', 'truck'],
    'detection_confidence': 0.6,
    'frame_delay': 0.3,
}

PERFORMANCE_CONFIG = {
    'max_detection_classes': 10,
    'image_resize': (320, 320),
    'frame_delay': 0.4,
}
```
