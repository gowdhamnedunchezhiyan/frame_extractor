import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import argparse
from typing import List, Tuple

class TruckExtractor:
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the truck extractor with YOLO model.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detection
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # COCO class IDs for vehicles that could be trucks
        self.truck_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
    def detect_trucks(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect trucks in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2, confidence)
        """
        results = self.model(frame, verbose=False)
        trucks = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter for truck-like vehicles with sufficient confidence
                    if class_id in self.truck_classes and confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        trucks.append((x1, y1, x2, y2, confidence))
        
        return trucks
    
    def crop_truck_image(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                        padding_ratio: float = 0.1) -> np.ndarray:
        """
        Crop truck image from frame with padding.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            padding_ratio: Extra padding around the truck
            
        Returns:
            Cropped truck image
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Add padding
        pad_w = int((x2 - x1) * padding_ratio)
        pad_h = int((y2 - y1) * padding_ratio)
        
        # Ensure coordinates stay within frame bounds
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return frame[y1:y2, x1:x2]
    
    def process_video(self, video_path: str, output_dir: str, 
                     frame_interval: int = 30, max_trucks_per_frame: int = 3,
                     min_truck_area: int = 5000):
        """
        Process video and extract truck images.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted images
            frame_interval: Extract trucks every N frames
            max_trucks_per_frame: Maximum trucks to extract per frame
            min_truck_area: Minimum area for a valid truck detection
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Extracting trucks every {frame_interval} frames")
        
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % frame_interval == 0:
                    trucks = self.detect_trucks(frame)
                    
                    # Sort by confidence and take top detections
                    trucks = sorted(trucks, key=lambda x: x[4], reverse=True)
                    trucks = trucks[:max_trucks_per_frame]
                    
                    for i, (x1, y1, x2, y2, conf) in enumerate(trucks):
                        # Check minimum area
                        area = (x2 - x1) * (y2 - y1)
                        if area < min_truck_area:
                            continue
                        
                        # Crop truck image
                        truck_img = self.crop_truck_image(frame, (x1, y1, x2, y2))
                        
                        # Save image
                        timestamp = frame_count / fps
                        filename = f"truck_{extracted_count:06d}_t{timestamp:.2f}_conf{conf:.2f}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        cv2.imwrite(filepath, truck_img)
                        extracted_count += 1
                        
                        print(f"Extracted truck {extracted_count}: {filename} "
                              f"(area: {area}, conf: {conf:.2f})")
                
                frame_count += 1
                
                # Progress update
                if frame_count % (frame_interval * 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                    
        finally:
            cap.release()
        
        print(f"\nExtraction complete!")
        print(f"Total trucks extracted: {extracted_count}")
        print(f"Images saved to: {output_dir}")
    
    def create_annotation_template(self, output_dir: str):
        """
        Create a simple annotation template for license plate labeling.
        """
        template_path = os.path.join(output_dir, "annotation_template.txt")
        
        with open(template_path, 'w') as f:
            f.write("# License Plate Annotation Template\n")
            f.write("# Format: filename.jpg x1,y1,x2,y2,license_plate_text\n")
            f.write("# Example: truck_000001_t12.34_conf0.85.jpg 150,200,250,230,ABC123\n\n")
            
            # List all extracted images
            image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
            for img_file in image_files:
                f.write(f"{img_file} \n")
        
        print(f"Annotation template created: {template_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract truck images from video for license plate training")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("-o", "--output", default="extracted_trucks", help="Output directory")
    parser.add_argument("-i", "--interval", type=int, default=30, help="Frame interval for extraction")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("-m", "--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--max-trucks", type=int, default=3, help="Max trucks per frame")
    parser.add_argument("--min-area", type=int, default=5000, help="Minimum truck area")
    parser.add_argument("--create-template", action="store_true", help="Create annotation template")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = TruckExtractor(args.model, args.confidence)
    
    # Process video
    extractor.process_video(
        args.video_path,
        args.output,
        args.interval,
        args.max_trucks,
        args.min_area
    )
    
    # Create annotation template if requested
    if args.create_template:
        extractor.create_annotation_template(args.output)


if __name__ == "__main__":
    main()


# Usage Examples:
"""
# Basic usage
python truck_extractor.py video.mp4

# Custom settings
python truck_extractor.py video.mp4 -o my_trucks -i 15 -c 0.7 --max-trucks 5

# Create annotation template
python truck_extractor.py video.mp4 --create-template

# Use custom YOLO model
python truck_extractor.py video.mp4 -m yolov8s.pt

# Programmatic usage
extractor = TruckExtractor()
extractor.process_video("traffic_video.mp4", "output_trucks/")
"""