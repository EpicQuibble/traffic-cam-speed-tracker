import cv2
import time
import threading
import queue
import os
import numpy as np
from collections import defaultdict, deque
#
##START  37.81132417716535, -122.25000569931613
#to 
# END 37.81013809467633, -122.24822269607837
#
#
#
#
# Stream and Header Configuration
STREAM_URL = 'https://wzmedia.dot.ca.gov/D4/E580_at_Grand_Lakeshore.stream/playlist.m3u8'
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"
REFERER = "https://trafficvision.live/"
ORIGIN = "https://trafficvision.live"

# Real-world distance we can see in the camera view
# Measured from GPS coordinates at bottom vs top of visible highway
TOTAL_DISTANCE_FEET = 400

# Thread-safe Queue for Frames
frame_queue = queue.Queue(maxsize=450)
stop_event = threading.Event()

# Vehicle tracking data structures
tracked_vehicles = {}  # stores all our active vehicle trackers
next_vehicle_id = 0
vehicle_id_lock = threading.Lock()

# Try to use YOLO if available, otherwise fall back to background subtraction
USE_YOLO = False
try:
    from ultralytics import YOLO
    print("Attempting to load YOLOv8 model...")
    yolo_model = YOLO('yolov8n.pt')  # Nano model fastest and lightest
    USE_YOLO = True
    print("YOLOv8 loaded successfully!")
except:
    print("YOLO not available, using background subtraction method.")
    print("To use YOLO: pip install ultralytics")
    # Background subtractor helps us find moving objects by learning what the "background" looks like
    # Then anything that changes = probably a vehicle
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=25, detectShadows=True
    )

# Define where our timing lines should be (as fraction of screen height, 0=top, 1=bottom)
# These mark the start and end of our measurement zone
START_LINE_Y = 0.95  # Near bottom - where cars enter our view clearly
END_LINE_Y = 0.15    # Higher up - where we can still see cars well before perspective gets weird

class VehicleTracker:
    """
    Tracks a single vehicle as it moves through the frame.
    Stores its position history and calculates speed based on how far it travels vertically.
    """
    def __init__(self, vehicle_id, initial_pos, timestamp, frame_height):
        self.id = vehicle_id
        # Keep last 15 positions for smooth trajectory lines (not too many = less squiggly)
        self.positions = deque(maxlen=15)
        self.timestamps = deque(maxlen=15)
        # Track y-positions separately since vertical movement = moving toward/away from camera
        self.y_positions = deque(maxlen=15)
        self.last_seen = timestamp
        # Random color so we can tell vehicles apart
        self.color = tuple(np.random.randint(50, 255, 3).tolist())
        self.frame_height = frame_height
        
        # Store first detection
        self.positions.append(initial_pos)
        self.timestamps.append(timestamp)
        self.y_positions.append(initial_pos[1])
        
    def update(self, pos, timestamp):
        """
        Add a new position for this vehicle.
        Returns False if the position seems wrong (jumped too far = probably not the same car)
        """
        # Sanity check - did the vehicle jump super far? If so, probably a false match
        if len(self.positions) > 0:
            last_pos = self.positions[-1]
            distance = np.sqrt((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)
            
            # 100 pixels is reasonable for one frame, more than that = probably wrong
            if distance > 100:
                return False
        
        self.positions.append(pos)
        self.timestamps.append(timestamp)
        self.y_positions.append(pos[1])
        self.last_seen = timestamp
        return True
    
    def get_speed_mph(self):
        """
        Calculate speed based on how much vertical distance (y-axis) the car traveled.
        Vertical movement in the image = car moving along the highway toward/away from camera.
        
        The key insight: if a car moves from y=800 to y=400 (up the screen),
        it traveled some real-world distance that we can map to our 682 feet.
        """
        # Need at least a few data points to calculate speed reliably
        if len(self.y_positions) < 5:
            return 0
        
        # Find how far the car moved vertically (in pixels)
        min_y = min(self.y_positions)  # Highest point on screen (smallest y value)
        max_y = max(self.y_positions)  # Lowest point on screen (largest y value)
        vertical_travel_pixels = max_y - min_y
        
        # Make sure car moved enough to get a good measurement
        # If it only moved like 5% of the screen, speed calculation will be noisy
        if vertical_travel_pixels < self.frame_height * 0.12:
            return 0
        
        # Find when the car was at those positions to calculate time elapsed
        min_y_idx = list(self.y_positions).index(min_y)
        max_y_idx = list(self.y_positions).index(max_y)
        time_elapsed = abs(self.timestamps[max_y_idx] - self.timestamps[min_y_idx])
        
        # Too fast? Probably noise or bad detection
        if time_elapsed < 0.2:
            return 0
        
        # map screen pixels to real-world feet
        # Our measurement zone is between START_LINE (0.85) and END_LINE (0.25)
        # That's 60% of the screen height = 682 feet in real world
        measurement_zone_fraction = START_LINE_Y - END_LINE_Y  # 0.60
        
        # What fraction of the measurement zone did the car travel?
        fraction_of_zone = vertical_travel_pixels / (self.frame_height * measurement_zone_fraction)
        
        # Convert to real-world distance
        distance_traveled_feet = TOTAL_DISTANCE_FEET * fraction_of_zone
        
        # Calculate speed: distance / time
        feet_per_second = distance_traveled_feet / time_elapsed
        
        # Convert feet per second to MPH (multiply by 0.681818)
        mph = feet_per_second * 0.681818
        
        # Sanity check - highway speeds should be 30-90 mph realistically
        return max(0, min(mph, 95))
    
    def is_stale(self, current_time, timeout=1.5):
        """Check if we haven't seen this vehicle in a while (probably left the frame)"""
        return (current_time - self.last_seen) > timeout


def detect_vehicles_background_subtraction(frame, frame_count):
    """
    Fallback detection method using background subtraction.
    This learns what the "static" background looks like, then flags anything that moves.
    Not as accurate as YOLO but works without any ML models.
    """
    # Apply background subtraction - this gives us a mask of "foreground" pixels
    fg_mask = bg_subtractor.apply(frame, learningRate=0.001)
    
    # Background subtractor marks shadows as gray (127), we only want true foreground (255)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Clean up noise - morphological operations help merge nearby blobs and remove tiny spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fill gaps
    
    # Extra blur to merge close detections
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find blobs in the mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vehicles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size - too small = noise, too big = probably not a single vehicle
        if area < 500 or area > 12000:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio - vehicles should be roughly rectangular, not super thin or tall
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 4.0:
            continue
        
        # Ignore stuff right at the edges - usually noise or partially visible objects
        if y < 20 or y > frame.shape[0] - 20:
            continue
        
        vehicles.append((x, y, w, h))
    
    return vehicles

def match_vehicle_to_tracker(detection, trackers, current_time, threshold=100):
    """
    Try to match a new detection to an existing vehicle tracker.
    We look for the closest tracker and predict where it should be based on velocity.
    """
    # Center point of the detected vehicle
    det_x = detection[0] + detection[2] // 2
    det_y = detection[1] + detection[3] // 2
    det_center = (det_x, det_y)
    
    best_match = None
    best_score = float('inf')
    
    for vid, tracker in trackers.items():
        # Skip trackers that haven't been updated recently
        if tracker.is_stale(current_time, timeout=0.5):
            continue
        
        if len(tracker.positions) > 0:
            last_pos = tracker.positions[-1]
            
            # Basic distance to last known position
            distance = np.sqrt((det_center[0] - last_pos[0])**2 + 
                             (det_center[1] - last_pos[1])**2)
            
            # Better idea: predict where the vehicle SHOULD be based on its velocity
            if len(tracker.positions) >= 2:
                # Calculate velocity from last two positions
                velocity_x = tracker.positions[-1][0] - tracker.positions[-2][0]
                velocity_y = tracker.positions[-1][1] - tracker.positions[-2][1]
                
                # Predict next position (multiply by 3 since we process every few frames)
                predicted_x = last_pos[0] + velocity_x * 3
                predicted_y = last_pos[1] + velocity_y * 3
                predicted_pos = (predicted_x, predicted_y)
                
                # Distance to predicted position
                pred_distance = np.sqrt((det_center[0] - predicted_pos[0])**2 + 
                                      (det_center[1] - predicted_pos[1])**2)
                
                # Use whichever is closer - actual or predicted
                distance = min(distance, pred_distance)
            
            # Keep track of best match
            if distance < threshold and distance < best_score:
                best_score = distance
                best_match = vid
    
    return best_match

def draw_annotations(frame, trackers, frame_height):
    """
    Draw center dots, trajectory lines, and speed labels.
    """
    annotated = frame.copy()
    
    # Draw our measurement lines
    start_y = int(frame_height * START_LINE_Y)
    cv2.line(annotated, (0, start_y), (annotated.shape[1], start_y), (255, 0, 0), 3)
    cv2.putText(annotated, "START", (10, start_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    end_y = int(frame_height * END_LINE_Y)
    cv2.line(annotated, (0, end_y), (annotated.shape[1], end_y), (0, 0, 255), 3)
    cv2.putText(annotated, "END", (10, end_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Track stats for summary
    active_count = 0
    total_speed = 0
    speed_count = 0
    
    # Draw each tracked vehicle as a dot + trajectory + speed
    for vid, tracker in list(trackers.items()):
        if len(tracker.positions) > 0:
            # Get current position
            last_pos = tracker.positions[-1]
            x, y = int(last_pos[0]), int(last_pos[1])
            
            # Draw a nice big dot for the vehicle center
            cv2.circle(annotated, (x, y), 8, tracker.color, -1)  # Filled circle
            cv2.circle(annotated, (x, y), 8, (255, 255, 255), 2)  # White outline
            
            # Draw trajectory line (only use every other point for smoothness)
            if len(tracker.positions) > 2:
                points = np.array([[int(p[0]), int(p[1])] 
                                 for i, p in enumerate(tracker.positions) 
                                 if i % 2 == 0], np.int32)
                if len(points) > 1:
                    cv2.polylines(annotated, [points], False, tracker.color, 2)
            
            # Calculate and show speed
            speed = tracker.get_speed_mph()
            if speed > 5:  # Only show speed if car is actually moving
                active_count += 1
                total_speed += speed
                speed_count += 1
                
                # Draw speed label with a background box so it's readable
                speed_text = f"{speed:.0f} mph"
                text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background rectangle
                cv2.rectangle(annotated, 
                            (x - text_size[0]//2 - 4, y - 30), 
                            (x + text_size[0]//2 + 4, y - 8), 
                            tracker.color, -1)
                # Text
                cv2.putText(annotated, speed_text, 
                           (x - text_size[0]//2, y - 14), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Summary stats in top-left corner
    avg_speed = total_speed / speed_count if speed_count > 0 else 0
    
    # Black background box
    cv2.rectangle(annotated, (5, 5), (320, 95), (0, 0, 0), -1)
    cv2.rectangle(annotated, (5, 5), (320, 95), (0, 255, 0), 2)
    
    cv2.putText(annotated, f"Vehicles: {active_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, f"Avg Speed: {avg_speed:.1f} mph", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, f"Detection: {'YOLO' if USE_YOLO else 'Background Sub'}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return annotated

def stream_reader_thread():
    """
    This runs in a separate thread and just reads frames from the stream continuously.
    Puts them in a queue for the main thread to process.
    """
    print("Reader thread started.")
    
    try:
        # Set up headers so the stream server accepts our requests
        headers = f"User-Agent: {USER_AGENT}\r\nReferer: {REFERER}\r\nOrigin: {ORIGIN}\r\n"
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'headers\n{headers}\n'
        
        cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print(f"Reader thread: Error - Could not open stream: {STREAM_URL}")
            return

        print("Reader thread: Stream opened successfully.")

        while not stop_event.is_set():
            ret, frame = cap.read()
            
            if not ret:
                # Stream dropped, try to reconnect
                print("Reader thread: Stream ended, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    print("Reader thread: Failed to reopen, retrying in 5s...")
                    time.sleep(5)
                
                continue
            
            # If queue is full, drop oldest frame to stay current
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            frame_queue.put(frame)
            
    except Exception as e:
        print(f"Reader thread: Error: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print("Reader thread stopped.")

def start_stream_processing():
    """
    Main processing loop - pulls frames from queue, runs detection, tracks vehicles, draws output.
    """
    global next_vehicle_id, tracked_vehicles
    
    # Start background thread to read stream
    reader_thread = threading.Thread(target=stream_reader_thread)
    reader_thread.daemon = True 
    reader_thread.start()

    print("Main thread: Waiting for stream to start...")
    time.sleep(2)
    
    # Build up a small buffer before we start processing
    print("Main thread: Building buffer...")
    initial_buffer_target = 150
    while frame_queue.qsize() < initial_buffer_target and reader_thread.is_alive():
        print(f"Main thread: Buffer at {frame_queue.qsize()}/{initial_buffer_target} frames...")
        time.sleep(1)
    
    if not reader_thread.is_alive():
        print("Main thread: Reader thread died during startup.")
        cv2.destroyAllWindows()
        return

    print("Main thread: Starting processing!")
    if not USE_YOLO:
        print("Background model will stabilize in ~15 seconds...")

    TARGET_FPS = 30
    FRAME_DURATION = 1.0 / TARGET_FPS
    # Process every 2nd frame with YOLO, every 4th with background subtraction
    PROCESS_EVERY_N_FRAMES = 2 if USE_YOLO else 4
    
    frame_count = 0

    while True:
        loop_start_time = time.time()

        try:
            frame = frame_queue.get(timeout=5.0)
            frame_count += 1
            current_time = time.time()
            
            # Run detection every N frames to save CPU
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Detect vehicles in current frame
                if USE_YOLO:
                    detections = detect_vehicles_yolo(frame)
                else:
                    detections = detect_vehicles_background_subtraction(frame, frame_count)
                
                # Match each detection to existing trackers
                matched_ids = set()
                
                for detection in detections:
                    # Try to find which existing vehicle this detection belongs to
                    matched_id = match_vehicle_to_tracker(detection, tracked_vehicles, current_time)
                    
                    x, y, w, h = detection
                    center = (x + w // 2, y + h // 2)
                    
                    if matched_id is not None:
                        # Update existing tracker
                        success = tracked_vehicles[matched_id].update(center, current_time)
                        if success:
                            matched_ids.add(matched_id)
                    else:
                        # New vehicle - create a tracker for it
                        with vehicle_id_lock:
                            new_id = next_vehicle_id
                            next_vehicle_id += 1
                        
                        tracked_vehicles[new_id] = VehicleTracker(
                            new_id, center, current_time, frame.shape[0]
                        )
                        matched_ids.add(new_id)
                
                # Clean up trackers we haven't seen in a while
                stale_ids = [vid for vid, tracker in tracked_vehicles.items() 
                           if tracker.is_stale(current_time)]
                for vid in stale_ids:
                    del tracked_vehicles[vid]
            
            # Draw visualization (every frame, even if we didn't detect on this frame)
            annotated_frame = draw_annotations(frame, tracked_vehicles, frame.shape[0])
            
            # Show it
            display_frame = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow('Traffic Speed Monitor (Press "q" to quit)', display_frame)

            # Frame rate limiting
            elapsed_time = time.time() - loop_start_time
            wait_time = FRAME_DURATION - elapsed_time
            key_wait_ms = 1 
            
            if wait_time > 0.001:
                key_wait_ms = int(wait_time * 1000)

            key = cv2.waitKey(key_wait_ms)
            if key & 0xFF == ord('q'):
                print("Quitting...")
                stop_event.set()
                break

        except queue.Empty:
            if not reader_thread.is_alive():
                print("Main thread: Reader died. Exiting.")
                break
            print("Main thread: Waiting for frames...")
            continue
        
        except Exception as e:
            print(f"Main thread: Error: {e}")
            import traceback
            traceback.print_exc()
            stop_event.set()
            break
    
    # Cleanup
    print("Shutting down...")
    reader_thread.join(timeout=5)
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    start_stream_processing()