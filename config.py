import numpy as np

# --- Stream Configuration ---
STREAM_URL = 'https://wzmedia.dot.ca.gov/D4/E580_at_Grand_Lakeshore.stream/playlist.m3u8'
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"
REFERER = "me!" 
ORIGIN = "myself" 

# Real-world distance in feet between the START_ZONE and END_ZONE
TOTAL_DISTANCE_FEET = 400 # BADDDD ESTIMATTE

# Set to False to force background subtraction
USE_YOLO = True 
# YOLO classes for [car, motorcycle, bus, truck]
VEHICLE_CLASSES = [2, 3, 5, 7]  #COCO class IDs
YOLO_MODEL = 'yolov8n.pt'  # Path to YOLO model or model name

# How many frames to skip between detections (higher = faster but less accurate)
PROCESS_EVERY_N_FRAMES_YOLO = 2 #NEED TO GET FPS HIGHER lower this 
PROCESS_EVERY_N_FRAMES_BGSUB = 4

TARGET_FPS = 30 #NEED TO GET FPS HIGHER

SHOW_DEBUG_GRID = True  # Set to True to see a coordinate grid

# We define polygons (zones) to match the road's perspective.
# These are (x, y) pairs from the top-left corner.

# START_ZONE (bottom, where cars enter)
START_ZONE_FRAC = np.array([
    [0.2,0.8],  
    [0.05, 0.45], 
    [0.05, 0.95],
    [0.75, 0.95] 
], dtype=np.float32)

# END_ZONE (top , where cars exit)
END_ZONE_FRAC = np.array([
    [0.9, 0.1],  
    [0.7, 0.1], 
    [0.7, 0.12],   
    [0.94, 0.12]    
], dtype=np.float32)