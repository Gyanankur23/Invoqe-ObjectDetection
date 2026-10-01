"""
Object Tracking using SORT (Simple Online and Realtime Tracking)
Implements object tracking with unique IDs
"""

import cv2
import numpy as np
from detection import ObjectDetector
from typing import List, Dict, Optional
from scipy.optimize import linear_sum_assignment

class Track:
    """Track object for storing tracking information"""
    
    def __init__(self, track_id: int, bbox: List[int], class_name: str, max_age: int = 30):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.age = 0
        self.max_age = max_age
        self.hits = 1
        self.time_since_update = 0
    
    def update(self, bbox: List[int]):
        """Update track with new bounding box"""
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
    
    def increment_age(self):
        """Increment age of track"""
        self.age += 1
        self.time_since_update += 1
    
    def is_dead(self) -> bool:
        """Check if track should be removed"""
        return self.time_since_update > self.max_age

class ObjectTracker:
    """Object tracker using simple IOU matching"""
    
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
    
    def iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """Update tracks with new detections"""
        if not detections:
            # No detections, increment age of all tracks
            for track in self.tracks:
                track.increment_age()
            # Remove dead tracks
            self.tracks = [t for t in self.tracks if not t.is_dead()]
            return self.tracks
        
        if not self.tracks:
            # No existing tracks, create new ones
            for det in detections:
                track = Track(
                    self.next_track_id,
                    det['bbox'],
                    det['class_name'],
                    self.max_age
                )
                self.tracks.append(track)
                self.next_track_id += 1
            return self.tracks
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.iou(track.bbox, det['bbox'])
        
        # Hungarian algorithm for assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        # Matched pairs
        matched_tracks = set()
        matched_detections = set()
        
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= self.iou_threshold:
                self.tracks[i].update(detections[j]['bbox'])
                matched_tracks.add(i)
                matched_detections.add(j)
        
        # Unmatched tracks (increment age)
        for i in range(len(self.tracks)):
            if i not in matched_tracks:
                self.tracks[i].increment_age()
        
        # Unmatched detections (create new tracks)
        for j in range(len(detections)):
            if j not in matched_detections:
                track = Track(
                    self.next_track_id,
                    detections[j]['bbox'],
                    detections[j]['class_name'],
                    self.max_age
                )
                self.tracks.append(track)
                self.next_track_id += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        
        return self.tracks
    
    def draw_tracks(self, image: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw tracks on image"""
        # Color map for different track IDs
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id
            class_name = track.class_name
            
            # Select color based on track ID
            color = colors[track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with track ID
            label = f"ID: {track_id} {class_name}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image

class DetectionAndTracking:
    """Combined detection and tracking system"""
    
    def __init__(self, model_name='yolov8n.pt', max_age=30, iou_threshold=0.3):
        self.detector = ObjectDetector(model_name)
        self.tracker = ObjectTracker(max_age, iou_threshold)
    
    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Track]]:
        """Process frame with detection and tracking"""
        # Detect objects
        _, detections = self.detector.detect_frame(frame, conf_threshold)
        
        # Update tracks
        tracks = self.tracker.update(detections)
        
        # Draw tracks
        annotated_frame = self.tracker.draw_tracks(frame.copy(), tracks)
        
        return annotated_frame, tracks
    
    def process_video(self, video_path: str, output_path: str = None, conf_threshold: float = 0.5):
        """Process video with detection and tracking"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print(f"Processing video with tracking: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, tracks = self.process_frame(frame, conf_threshold)
            
            # Draw track count
            cv2.putText(annotated_frame, f"Tracks: {len(tracks)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame
            if output_path:
                out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"Video processing complete. Total frames: {frame_count}")
        if output_path:
            print(f"Output saved to: {output_path}")
    
    def process_webcam(self, conf_threshold: float = 0.5):
        """Real-time detection and tracking from webcam"""
        import time
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        print("Starting webcam detection and tracking. Press 'q' to quit.")
        
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
            
            # Process frame
            annotated_frame, tracks = self.process_frame(frame, conf_threshold)
            
            # Draw FPS
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw track count
            cv2.putText(annotated_frame, f"Tracks: {len(tracks)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Object Detection & Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam detection stopped.")

def main():
    """Test detection and tracking"""
    system = DetectionAndTracking('yolov8n.pt')
    system.process_webcam(conf_threshold=0.5)

if __name__ == "__main__":
    main()
