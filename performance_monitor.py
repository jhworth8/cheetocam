#!/usr/bin/env python3
"""
Performance monitoring script for YOLOv8 on Raspberry Pi 5
This script helps optimize detection performance by monitoring CPU, memory, and inference times.
"""

import time
import psutil
import cv2
from ultralytics import YOLO
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_system_resources():
    """Monitor current system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    temperature = None
    
    try:
        # Try to get CPU temperature (Raspberry Pi specific)
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp_millicelsius = int(f.read())
            temperature = temp_millicelsius / 1000
    except:
        pass
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_mb': memory.available / (1024 * 1024),
        'temperature_celsius': temperature
    }

def benchmark_yolo_model(model_path='yolov8n.pt', test_image_size=(640, 640), num_iterations=10):
    """Benchmark YOLOv8 model performance."""
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create a dummy image for testing
    dummy_image = np.random.randint(0, 255, (test_image_size[1], test_image_size[0], 3), dtype=np.uint8)
    
    logger.info(f"Running {num_iterations} inference iterations...")
    inference_times = []
    
    for i in range(num_iterations):
        start_time = time.time()
        
        # Run inference
        results = model(dummy_image, verbose=False)
        
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        if i % 5 == 0:
            logger.info(f"Iteration {i+1}/{num_iterations}: {inference_time:.3f}s")
    
    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    std_inference_time = np.std(inference_times)
    
    return {
        'avg_inference_time': avg_inference_time,
        'min_inference_time': min_inference_time,
        'max_inference_time': max_inference_time,
        'std_inference_time': std_inference_time,
        'fps': 1.0 / avg_inference_time,
        'all_times': inference_times
    }

def optimize_camera_settings():
    """Test different camera resolutions for optimal performance."""
    cap = cv2.VideoCapture("/dev/video0")
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return None
    
    resolutions = [
        (320, 240),   # Low resolution
        (640, 480),   # Medium resolution
        (1280, 720),  # High resolution
    ]
    
    results = {}
    
    for width, height in resolutions:
        logger.info(f"Testing resolution: {width}x{height}")
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verify resolution was set
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Test frame capture time
        capture_times = []
        for _ in range(10):
            start_time = time.time()
            ret, frame = cap.read()
            end_time = time.time()
            
            if ret:
                capture_times.append(end_time - start_time)
        
        if capture_times:
            avg_capture_time = np.mean(capture_times)
            results[f"{actual_width}x{actual_height}"] = {
                'avg_capture_time': avg_capture_time,
                'fps': 1.0 / avg_capture_time
            }
    
    cap.release()
    return results

def main():
    """Main performance monitoring function."""
    logger.info("Starting YOLOv8 performance monitoring on Raspberry Pi 5")
    
    # Monitor system resources
    logger.info("=== System Resources ===")
    resources = monitor_system_resources()
    logger.info(f"CPU Usage: {resources['cpu_percent']:.1f}%")
    logger.info(f"Memory Usage: {resources['memory_percent']:.1f}%")
    logger.info(f"Available Memory: {resources['memory_available_mb']:.1f} MB")
    if resources['temperature_celsius']:
        logger.info(f"CPU Temperature: {resources['temperature_celsius']:.1f}°C")
    
    # Benchmark YOLOv8 model
    logger.info("\n=== YOLOv8 Model Benchmark ===")
    benchmark_results = benchmark_yolo_model()
    logger.info(f"Average Inference Time: {benchmark_results['avg_inference_time']:.3f}s")
    logger.info(f"Min Inference Time: {benchmark_results['min_inference_time']:.3f}s")
    logger.info(f"Max Inference Time: {benchmark_results['max_inference_time']:.3f}s")
    logger.info(f"Standard Deviation: {benchmark_results['std_inference_time']:.3f}s")
    logger.info(f"Estimated FPS: {benchmark_results['fps']:.1f}")
    
    # Test camera settings
    logger.info("\n=== Camera Performance Test ===")
    camera_results = optimize_camera_settings()
    if camera_results:
        logger.info("Camera performance by resolution:")
        for resolution, metrics in camera_results.items():
            logger.info(f"  {resolution}: {metrics['fps']:.1f} FPS")
    
    # Recommendations
    logger.info("\n=== Performance Recommendations ===")
    
    if benchmark_results['fps'] < 5:
        logger.warning("Low FPS detected. Consider:")
        logger.warning("- Reducing input resolution")
        logger.warning("- Increasing FRAME_DELAY in .env")
        logger.warning("- Using YOLOv8n instead of larger models")
    
    if resources['cpu_percent'] > 80:
        logger.warning("High CPU usage detected. Consider:")
        logger.warning("- Reducing detection frequency")
        logger.warning("- Optimizing other running processes")
    
    if resources['temperature_celsius'] and resources['temperature_celsius'] > 70:
        logger.warning("High CPU temperature detected. Consider:")
        logger.warning("- Adding cooling")
        logger.warning("- Reducing CPU load")
    
    logger.info("Performance monitoring complete!")

if __name__ == "__main__":
    main()
