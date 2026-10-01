"""
Object Detection using YOLO (Ultralytics)
Supports image, video, and webcam detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
from typing import List, Dict, Tuple
import time

class ObjectDetector:
    """Object detection using YOLO model"""
    
    def __init__(self, model_name='yolov8n.pt'):
        """Initialize YOLO model"""
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.class_names = self.model.names
        print(f"Model loaded. Classes: {len(self.class_names)}")
    
    def detect_image(self, image_path: str, conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
        """Detect objects in an image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run detection
        results = self.model(image, conf=conf_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        # Draw detections on image
        annotated_image = self.draw_detections(image.copy(), detections)
        
        return annotated_image, detections
    
    def detect_frame(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
        """Detect objects in a video frame"""
        # Run detection
        results = self.model(frame, conf=conf_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        # Draw detections
        annotated_frame = self.draw_detections(frame.copy(), detections)
        
        return annotated_frame, detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image
    
    def detect_video(self, video_path: str, output_path: str = None, conf_threshold: float = 0.5):
        """Detect objects in video file"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            annotated_frame, detections = self.detect_frame(frame, conf_threshold)
            total_detections += len(detections)
            frame_count += 1
            
            # Write frame
            if output_path:
                out.write(annotated_frame)
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        
        print(f"Video processing complete. Total frames: {frame_count}")
        print(f"Total detections: {total_detections}")
        
        if output_path:
            print(f"Output saved to: {output_path}")
    
    def detect_webcam(self, conf_threshold: float = 0.5):
        """Real-time detection from webcam"""
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        print("Starting webcam detection. Press 'q' to quit.")
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = time.time()
                current_fps = fps_counter / (fps_end_time - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Detect objects
            annotated_frame, detections = self.detect_frame(frame, conf_threshold)
            
            # Draw FPS
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw object count
            cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Object Detection', annotated_frame)
            
            # Quit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam detection stopped.")

def main():
    """Test the object detector"""
    detector = ObjectDetector('yolov8n.pt')
    
    # Test with webcam
    detector.detect_webcam(conf_threshold=0.5)

if __name__ == "__main__":
    main()
