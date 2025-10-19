# YOLOv8 Upgrade for Raspberry Pi 5

This upgrade replaces the old YOLOv3-tiny model with the modern YOLOv8n model, providing:

- **80 object classes** (vs 80 in YOLOv3, but with better accuracy)
- **Better detection accuracy** and speed
- **Smaller model size** (~6MB vs ~33MB for YOLOv3-tiny)
- **Modern architecture** with improved performance
- **Multi-class detection** support

## What's New

### Enhanced Detection Capabilities
- **Primary focus**: Cats (maintained from original)
- **Additional classes**: Dogs, people, birds, vehicles, and more
- **Configurable detection**: Set `DETECTION_CLASSES` in your `.env` file
- **Better accuracy**: YOLOv8 provides superior detection performance

### Performance Improvements
- **Faster inference**: Optimized for Raspberry Pi 5
- **Lower memory usage**: More efficient model architecture
- **Better real-time performance**: Improved frame processing

## Installation

### 1. Install Dependencies
```bash
# Make the installation script executable
chmod +x install_yolov8.sh

# Run the installation script
./install_yolov8.sh
```

### 2. Configure Environment
```bash
# Copy the example configuration
cp env_example.txt .env

# Edit your configuration
nano .env
```

### 3. Key Configuration Options

#### Detection Settings
```bash
# Enable multi-class detection
ENABLE_MULTI_CLASS_DETECTION=1

# Detection confidence threshold (0.0-1.0)
DETECTION_CONFIDENCE=0.5

# Classes to detect (comma-separated)
DETECTION_CLASSES=cat,dog,person,bird,car,truck,bicycle,motorbike,bus,train

# Frame processing delay (seconds)
FRAME_DELAY=0.2
```

#### Available Detection Classes
The YOLOv8 model can detect 80 different object classes including:
- **Animals**: cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- **People**: person
- **Vehicles**: car, truck, bus, train, bicycle, motorbike, boat, aeroplane
- **Objects**: bottle, cup, chair, laptop, cell phone, book, clock, etc.

## Usage

### Basic Usage
```bash
# Run the upgraded detector
python3 cat_detector_yolo_v8.py
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
1. **Use YOLOv8n**: The nano version is optimized for edge devices
2. **Adjust resolution**: Lower camera resolution for better performance
3. **Tune confidence**: Higher confidence = fewer false positives
4. **Frame delay**: Increase `FRAME_DELAY` if CPU usage is high

### Recommended Settings for Pi 5
```bash
# Optimal settings for Raspberry Pi 5
DETECTION_CONFIDENCE=0.6
FRAME_DELAY=0.3
DETECTION_CLASSES=cat,dog,person,bird
```

## Troubleshooting

### Model Download Issues
If the YOLOv8n model fails to download:
```bash
# Manual download
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Performance Issues
1. **Run performance monitor**: `python3 performance_monitor.py`
2. **Check CPU temperature**: Should be < 70°C
3. **Monitor memory usage**: Should have > 100MB available
4. **Adjust settings**: Increase `FRAME_DELAY` or reduce `DETECTION_CLASSES`

### Fallback to YOLOv3
If YOLOv8 doesn't work, the system automatically falls back to YOLOv3-tiny:
- Keep your existing `yolo/` directory
- The system will detect and use the fallback automatically

## File Structure
```
cheetocam/
├── cat_detector_yolo_v8.py      # New YOLOv8 detector
├── cat_detector_yolo_gemini.py   # Original detector (backup)
├── cat_detector_yolo.py          # Original detector (backup)
├── performance_monitor.py        # Performance monitoring
├── install_yolov8.sh            # Installation script
├── cat-detector.service         # Systemd service file
├── env_example.txt              # Configuration template
├── requirements.txt             # Updated dependencies
└── yolo/                        # Legacy YOLOv3 files (fallback)
    ├── yolov3-tiny.weights
    ├── yolov3-tiny.cfg
    └── coco.names
```

## Migration from YOLOv3

### Automatic Migration
The new system automatically:
1. Tries to load YOLOv8n first
2. Falls back to YOLOv3-tiny if YOLOv8 fails
3. Maintains all existing functionality
4. Adds new multi-class detection capabilities

### Configuration Changes
- `DETECTION_CLASSES`: New option for multi-class detection
- `DETECTION_CONFIDENCE`: More precise confidence control
- `ENABLE_MULTI_CLASS_DETECTION`: Enable/disable multi-class mode

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

| Model | Size | Classes | Speed (Pi 5) | Accuracy |
|-------|------|---------|--------------|----------|
| YOLOv3-tiny | ~33MB | 80 | ~3 FPS | Good |
| YOLOv8n | ~6MB | 80 | ~8-12 FPS | Better |

The YOLOv8n model provides significantly better performance while using less memory and storage space.
