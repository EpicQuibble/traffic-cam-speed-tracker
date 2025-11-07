import cv2
import numpy as np
from collections import deque
import config

class VehicleTracker:
    """
    Tracks a single vehicle, checks if it passes through detection zones,
    and calculates its speed.
    """
    def __init__(self, vehicle_id, initial_pos, timestamp, start_zone_px, end_zone_px):
        self.id = vehicle_id
        self.positions = deque(maxlen=15)  # Store last 15 positions
        self.last_seen = timestamp
        self.color = tuple(np.random.randint(50, 255, 3).tolist()) # Random color for the dot
        
        # Zone definitions (in pixels)
        self.start_zone = start_zone_px
        self.end_zone = end_zone_px
        
        # Speed calculation state
        self.start_time = None
        self.end_time = None
        self.speed_mph = 0.0
        
        # Add first position
        self.positions.append(initial_pos)
        
    def update(self, pos, timestamp):
        """
        Add a new position for this vehicle and check if it's in a zone.
        """
        self.positions.append(pos)
        self.last_seen = timestamp
        
        # Check if point is inside our zones
        # We only care about the *center* of the vehicle
        
        # Begin, car is spotted, check for START zone entry
        if self.start_time is None:
             # cv2.pointPolygonTest returns > 0 if inside, 0 if on edge, < 0 if outside
            if cv2.pointPolygonTest(self.start_zone, pos, False) >= 0:
                self.start_time = timestamp
        
        # END, car is tracked and checked for END zone entry (only after it has hit the start zone)
        elif self.end_time is None:
            if cv2.pointPolygonTest(self.end_zone, pos, False) >= 0:
                self.end_time = timestamp
                self.calculate_speed()

    def calculate_speed(self):
        """
        Calculates speed in MPH based on time to travel TOTAL_DISTANCE_FEET.
        """
        if self.start_time and self.end_time:
            time_elapsed_seconds = self.end_time - self.start_time
            
            # Sanity check (e.g., must take at least 0.2 seconds)
            if time_elapsed_seconds > 0.2:
                feet_per_second = config.TOTAL_DISTANCE_FEET / time_elapsed_seconds
                
                # Convert feet per second to MPH (1 FPS = 0.681818 MPH) idk what this math is ngl thanks chatgpt
                self.speed_mph = feet_per_second * 0.681818
                
            else:
                self.speed_mph = 0.0 # Time was too short --> kinda dumb because if a car goes super fast AKA racing then it won't register but oh well problem for later 
    
    def is_stale(self, current_time, timeout=1.5):
        """Check if we haven't seen this vehicle in a while."""
        return (current_time - self.last_seen) > timeout

def detect_vehicles_yolo(yolo_model, frame):
    """
    Uses YOLOv8 to detect vehicles.
    Returns a list of bounding boxes in (x, y, w, h) format.
    """
    detections = []
    # verbose=False silences the console spam from YOLO
    results = yolo_model(frame, classes=config.VEHICLE_CLASSES, verbose=False)
    
    # Process results
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = box
        if conf > 0.4: # Confidence threshold
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            detections.append((x, y, w, h))
            
    return detections

def detect_vehicles_background_subtraction(bg_subtractor, frame):
    """
    Fallback detection method using background subtraction.
    Returns a list of bounding boxes in (x, y, w, h) format.
    """
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame, learningRate=0.001)
    
    # Threshold to remove shadows (which are gray)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vehicles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size
        if area < 500 or area > 12000:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 4.0:
            continue
        
        # Ignore edges
        if y < 20 or y > frame.shape[0] - 20:
            continue
            
        vehicles.append((x, y, w, h))
    
    return vehicles

def match_vehicle_to_tracker(detection, trackers, current_time, threshold=100):
    """
    Matches a new detection (x, y, w, h) to the closest existing tracker.
    """
    det_center = (detection[0] + detection[2] // 2, detection[1] + detection[3] // 2)
    
    best_match = None
    best_score = float('inf')
    
    for vid, tracker in trackers.items():
        if tracker.is_stale(current_time, timeout=0.5):
            continue
        
        if len(tracker.positions) > 0:
            last_pos = tracker.positions[-1]
            distance = np.sqrt((det_center[0] - last_pos[0])**2 + 
                             (det_center[1] - last_pos[1])**2)
            
            # Predict where the vehicle *should* be
            if len(tracker.positions) >= 2:
                velocity_x = tracker.positions[-1][0] - tracker.positions[-2][0]
                velocity_y = tracker.positions[-1][1] - tracker.positions[-2][1]
                
                # Predict based on velocity
                predicted_x = last_pos[0] + velocity_x 
                predicted_y = last_pos[1] + velocity_y
                predicted_pos = (predicted_x, predicted_y)
                
                pred_distance = np.sqrt((det_center[0] - predicted_pos[0])**2 + 
                                      (det_center[1] - predicted_pos[1])**2)
                
                # Use the smaller of the two distances (actual vs. predicted)
                distance = min(distance, pred_distance)
            
            if distance < threshold and distance < best_score:
                best_score = distance
                best_match = vid
    
    return best_match