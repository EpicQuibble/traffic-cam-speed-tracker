import numpy as np

# --- Stream Configuration ---
STREAM_URL = 'https://wzmedia.dot.ca.gov/D4/E580_at_Grand_Lakeshore.stream/playlist.m3u8'
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"
REFERER = "https://trafficvision.live/"
ORIGIN = "https://trafficvision.live"

# --- Speed Calculation ---
# Real-world distance in feet between the START_ZONE and END_ZONE
TOTAL_DISTANCE_FEET = 400

# --- Detection Configuration ---
# Set to False to force background subtraction
USE_YOLO = True 
# YOLO classes for [car, motorcycle, bus, truck]
VEHICLE_CLASSES = [2, 3, 5, 7] 
# How many frames to skip between detections (higher = faster but less accurate)
PROCESS_EVERY_N_FRAMES_YOLO = 2
PROCESS_EVERY_N_FRAMES_BGSUB = 4

# --- Target Display FPS ---
TARGET_FPS = 30

# --- Debugging ---
SHOW_DEBUG_GRID = True  # Set to True to see a coordinate grid

# --- Zone Definitions (Fractional Coordinates 0.0 - 1.0) ---
# We define polygons (zones) to match the road's perspective.
# These are (x, y) pairs from the top-left corner.
# I've made educated guesses, but you will need to tune these!

# START_ZONE (at the bottom of the frame, where cars enter)
START_ZONE_FRAC = np.array([
    [0.2,0.8],  
    [0.05, 0.45], 
    [0.05, 0.95],
    [0.75, 0.95] 
], dtype=np.float32)
# START_ZONE (at the bottom of the frame, where cars enter)

# END_ZONE (further up the road, where cars exit the measurement area)
END_ZONE_FRAC = np.array([
    [0.9, 0.1],  
    [0.7, 0.1], 
    [0.7, 0.12],   
    [0.94, 0.12]    
], dtype=np.float32)