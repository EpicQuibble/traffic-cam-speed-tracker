import cv2
import time
import threading
import queue
import os
import config

# Thread-safe Queue for Frames
frame_queue = queue.Queue(maxsize=450)
stop_event = threading.Event()

def stream_reader_thread():
    """
    Runs in a separate thread to continuously read frames from the stream.
    Puts frames into the thread-safe 'frame_queue' for the main thread to process.
    """
    print("Reader thread started.")
    
    try:
        # Set up headers so the stream server accepts our requests
        headers = f"User-Agent: {config.USER_AGENT}\r\nReferer: {config.REFERER}\r\nOrigin: {config.ORIGIN}\r\n"
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'headers\n{headers}\n'
        
        cap = cv2.VideoCapture(config.STREAM_URL, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print(f"Reader thread: Error - Could not open stream: {config.STREAM_URL}")
            return

        print("Reader thread: Stream opened successfully.")

        while not stop_event.is_set():
            ret, frame = cap.read()
            
            if not ret:
                # Stream dropped, try to reconnect
                print("Reader thread: Stream ended, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(config.STREAM_URL, cv2.CAP_FFMPEG)
                
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