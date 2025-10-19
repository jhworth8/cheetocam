#!/bin/bash

# Installation script for YOLOv11 on Raspberry Pi 5
# This script will install the necessary dependencies and download the YOLOv11n model

echo "Installing YOLOv11 dependencies for Raspberry Pi 5..."

# Update system packages
sudo apt update
sudo apt upgrade -y

# Install Python dependencies
pip3 install --upgrade pip
pip3 install ultralytics>=8.3.0
pip3 install opencv-python==4.10.0.84
pip3 install numpy==2.1.3
pip3 install pillow==11.0.0
pip3 install requests==2.32.3
pip3 install pytz==2024.2
pip3 install google-generativeai==0.8.3
pip3 install supabase==2.9.0

# Download YOLOv11n model (this will happen automatically on first run)
echo "YOLOv11n model will be downloaded automatically on first run"
echo "Model size: ~6MB (much smaller than YOLOv3-tiny)"

# Set up camera permissions
echo "Setting up camera permissions..."
sudo usermod -a -G video $USER
sudo usermod -a -G dialout $USER

# Create necessary directories
mkdir -p logs
mkdir -p detections

# Set up systemd service (optional)
echo "To run as a service, copy the systemd service file and enable it:"
echo "sudo cp cat-detector.service /etc/systemd/system/"
echo "sudo systemctl enable cat-detector.service"
echo "sudo systemctl start cat-detector.service"

echo "Installation complete!"
echo "Next steps:"
echo "1. Edit detector_config.py to configure your settings"
echo "2. Run: python3 cat_detector_yolo_gemini_yolo11.py"
echo "3. The YOLOv11n model will be downloaded automatically on first run"
