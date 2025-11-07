Traffic Speed Estimator

Goal: Watch a live traffic camera and estimate vehicle speeds.

main.py: Runs the main application loop.

config.py: Holds all settings (URL, zones, distance).

video_stream.py: Connects to and reads the video stream.

detection.py: Detects and tracks vehicles.

ui.py: Draws the dots and speed numbers on the video.

Usage: Install with pip install opencv-python numpy ultralytics and run with python main.py.