import cv2
import time
import threading
import queue
import os
# no longer need streamlink
# import streamlink
# from streamlink.options import Options


# Stream and Header Configuration
STREAM_URL = 'https://wzmedia.dot.ca.gov/D4/E580_at_Grand_Lakeshore.stream/playlist.m3u8'
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"
REFERER = "https://trafficvision.live/"
ORIGIN = "https://trafficvision.live"

#### Thread-safe Queue for Frames ###
# Set queue size to hold ~10-15 seconds of 30fps video
frame_queue = queue.Queue(maxsize=450)
stop_event = threading.Event()

# Resolve Stream URL with Streamlink
# (*** DEPRECATED ***)

# Stream Reader Thread
def stream_reader_thread():
    """
    Reads frames from the video stream in a separate thread.
    We pass headers to FFMPEG via an environment variable,
    allowing OpenCV's native HLS support to work.
    """
    print("Reader thread started.")
    
    try:
        # Set FFMPEG headers for all requests made by VideoCapture
        # The format is 'key\nvalue\n'. Note the trailing newlines.
        headers = f"User-Agent: {USER_AGENT}\r\nReferer: {REFERER}\r\nOrigin: {ORIGIN}\r\n"
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'headers\n{headers}\n'
        
        # Open the master playlist URL directly with OpenCV
        cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print(f"Reader thread: Error - Could not open initial stream: {STREAM_URL}")
            print("Check if FFMPEG backend is available for OpenCV (it should be).")
            return

        print("Reader thread: Stream opened successfully.")

        while not stop_event.is_set():
            ret, frame = cap.read()
            
            if not ret:
                print("Reader thread: Stream ended or error. Attempting to reopen...")
                cap.release()
                time.sleep(2) # Wait 2 seconds before retrying

                # Re-open the stream. Headers are still set in os.environ
                cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    print("Reader thread: Failed to reopen stream. Retrying in 5s...")
                    time.sleep(5)
                
                continue # Skip to the next loop iteration
            
            # If the queue is full, drop the oldest frame to stay recent
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Put the new frame into the queue
            frame_queue.put(frame)
            
    except Exception as e:
        print(f"Reader thread: An unhandled error occurred: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print("Reader thread stopped.")

# Main Stream Processing Function
def start_stream_processing():
    """
    Starts the reader thread and processes frames from the queue
    at a stable, rate-limited FPS.
    """
    
    # Start the stream reader thread.
    reader_thread = threading.Thread(target=stream_reader_thread)
    reader_thread.daemon = True 
    reader_thread.start()

    print("Main thread: Waiting for reader thread to start...")
    time.sleep(2) 
    
    ### PRE-STREAM BUFFERING LOGIC ##
    print("Main thread: Building initial 10-second video buffer...")
    initial_buffer_target = 300 # ~10 seconds at 30fps
    while frame_queue.qsize() < initial_buffer_target and reader_thread.is_alive():
        print(f"Main thread: Buffer at {frame_queue.qsize()}/{initial_buffer_target} frames...")
        time.sleep(1)
    
    if not reader_thread.is_alive():
        print("Main thread: Reader thread died while building initial buffer.")
        print("Check reader thread logs above.")
        cv2.destroyAllWindows()
        return

    print("Main thread: Initial buffer complete. Starting stream.")

    #### FPS RATE-LIMITING LOGIC ####
    TARGET_FPS = 30
    FRAME_DURATION = 1.0 / TARGET_FPS # Time in seconds for one frame (ie 0.0333s)

    while True:
        loop_start_time = time.time() # Record start time of this loop

        try:
            # Get a frame from our buffer
            frame = frame_queue.get(timeout=5.0) 

            #
            #  COMPUTER VISION PROCESSING GOES HERE
            #

            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow('Live Traffic Stream (Press "q" to quit)', display_frame)

            ##### Rate Limiting Calculation #####
            elapsed_time = time.time() - loop_start_time
            wait_time = FRAME_DURATION - elapsed_time
            
            # We must wait at least 1ms for cv2.waitKey to process GUI events
            key_wait_ms = 1 
            
            if wait_time > 0.001: # If we have time to spare
                key_wait_ms = int(wait_time * 1000) # Convert to milliseconds

            # Wait for the calculated time
            key = cv2.waitKey(key_wait_ms)
            if key & 0xFF == ord('q'):
                print("'q' pressed. Signaling reader thread to stop...")
                stop_event.set()
                break

        except queue.Empty:
            if not reader_thread.is_alive():
                print("Main thread: Reader thread has died. Exiting.")
                break
            # This will happen if the network is too slow to keep the buffer full
            print("Main thread: Frame queue is empty, waiting...")
            continue
        
        except Exception as e:
            print(f"Main thread: An error occurred: {e}")
            stop_event.set()
            break
            
    if stop_event.is_set():
        print("Main thread: Stop event received, shutting down.")

    reader_thread.join(timeout=5)
    
    cv2.destroyAllWindows()
    print("Stream processing stopped.")

# Run the script
if __name__ == "__main__":
    start_stream_processing()