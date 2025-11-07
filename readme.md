Traffic Speed Estimator

Goal: Watch a live traffic camera and estimate vehicle speeds.

main.py: Runs the main application loop.

config.py: Holds all settings (URL, zones, distance).

video_stream.py: Connects to and reads the video stream.

detection.py: Detects and tracks vehicles.

ui.py: Draws the dots and speed numbers on the video.

Usage: Install with pip install opencv-python numpy ultralytics and run with python main.py.


TODO: 
* I need to fix the speed calculation its quite wrong, the distance is probably the issue. 
* Need to improve detection
* describe the center divider, and ensure the split is done for the two lanes of traffic.
* improve FPS and stuttering, could be my laptop could be the code!


