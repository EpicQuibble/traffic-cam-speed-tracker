import cv2
import time
import threading
import queue
import numpy as np

# Import our custom modules
import config
from video_stream import stream_reader_thread, frame_queue, stop_event
from detection import (VehicleTracker, detect_vehicles_yolo, 
                       detect_vehicles_background_subtraction, match_vehicle_to_tracker)
from ui import draw_simple_ui, draw_debug_zones, draw_debug_grid

# --- Global Tracking Data ---
tracked_vehicles = {}
next_vehicle_id = 0
vehicle_id_lock = threading.Lock()

def main():
    global next_vehicle_id, tracked_vehicles

    # --- Initialize Detector ---
    yolo_model = None
    bg_subtractor = None
    USE_YOLO = False

    if config.USE_YOLO:
        try:
            from ultralytics import YOLO
            print("Attempting to load YOLOv8 model...")
            yolo_model = YOLO('yolov8n.pt')  # Nano model is fastest
            USE_YOLO = True
            print("YOLOv8 loaded successfully!")
        except ImportError:
            print("YOLO (ultralytics) not found, falling back to background subtraction.")
            print("To install: pip install ultralytics")
            USE_YOLO = False
            
    if not USE_YOLO:
        print("Using background subtraction method.")
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=True
        )
    
    PROCESS_EVERY_N_FRAMES = config.PROCESS_EVERY_N_FRAMES_YOLO if USE_YOLO else config.PROCESS_EVERY_N_FRAMES_BGSUB

    # --- Start Stream Reader ---
    reader_thread = threading.Thread(target=stream_reader_thread)
    reader_thread.daemon = True 
    reader_thread.start()

    print("Main thread: Waiting for stream to start...")
    time.sleep(2)
    
    print("Main thread: Building frame buffer...")
    while frame_queue.qsize() < 150 and reader_thread.is_alive():
        print(f"Main thread: Buffer at {frame_queue.qsize()}/150 frames...")
        time.sleep(1)
    
    if not reader_thread.is_alive():
        print("Main thread: Reader thread died during startup. Exiting.")
        cv2.destroyAllWindows()
        return

    # --- Get first frame to scale zones ---
    try:
        first_frame = frame_queue.get(timeout=5.0)
        frame_height, frame_width, _ = first_frame.shape
        print(f"Main thread: Frame dimensions: {frame_width}x{frame_height}")

        # Scale fractional zone coordinates to absolute pixel coordinates
        start_zone_px = (config.START_ZONE_FRAC * [frame_width, frame_height]).astype(np.int32)
        end_zone_px = (config.END_ZONE_FRAC * [frame_width, frame_height]).astype(np.int32)

    except queue.Empty:
        print("Main thread: Failed to get first frame from queue. Exiting.")
        stop_event.set()
        reader_thread.join()
        return
        
    print("Main thread: Starting processing loop!")
    if not USE_YOLO:
        print("Background model will stabilize in ~15 seconds...")

    FRAME_DURATION = 1.0 / config.TARGET_FPS
    frame_count = 0

    while True:
        loop_start_time = time.time()

        try:
            frame = frame_queue.get(timeout=5.0)
            frame_count += 1
            current_time = time.time()
            
            # --- Detection (runs every N frames) ---
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                if USE_YOLO:
                    detections = detect_vehicles_yolo(yolo_model, frame)
                else:
                    detections = detect_vehicles_background_subtraction(bg_subtractor, frame)
                
                matched_ids = set()
                
                for detection in detections:
                    matched_id = match_vehicle_to_tracker(detection, tracked_vehicles, current_time)
                    
                    x, y, w, h = detection
                    center = (x + w // 2, y + h // 2)
                    
                    if matched_id is not None:
                        # Update existing tracker
                        tracked_vehicles[matched_id].update(center, current_time)
                        matched_ids.add(matched_id)
                    else:
                        # Create new tracker for new vehicle
                        with vehicle_id_lock:
                            new_id = next_vehicle_id
                            next_vehicle_id += 1
                        
                        tracked_vehicles[new_id] = VehicleTracker(
                            new_id, center, current_time, start_zone_px, end_zone_px
                        )
                        matched_ids.add(new_id)
                
                # Clean up stale trackers
                stale_ids = [vid for vid, tracker in tracked_vehicles.items() 
                           if tracker.is_stale(current_time)]
                for vid in stale_ids:
                    del tracked_vehicles[vid]
            
            # --- Visualization (runs every frame) ---
            annotated_frame = frame.copy()
            
            # Draw the debug grid if enabled
            if config.SHOW_DEBUG_GRID:
                draw_debug_grid(annotated_frame)
            
            # Draw the simple UI (dots and numbers)
            draw_simple_ui(annotated_frame, tracked_vehicles)
            
            # Draw the debug zones (you can comment this out once you're happy)
            draw_debug_zones(annotated_frame, start_zone_px, end_zone_px)
            
            # Show the frame
            display_frame = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow('Traffic Speed Monitor (Press "q" to quit)', display_frame)

            # Frame rate limiting
            elapsed_time = time.time() - loop_start_time
            wait_time = FRAME_DURATION - elapsed_time
            key_wait_ms = max(1, int(wait_time * 1000))

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
    
    # --- Cleanup ---
    print("Shutting down...")
    reader_thread.join(timeout=5)
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()