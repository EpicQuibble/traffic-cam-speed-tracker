import cv2

def draw_simple_ui(frame, trackers):
    """
    Draws simple dot and number
    """
    for vid, tracker in trackers.items():
        if len(tracker.positions) > 0:
            # Get current position
            last_pos = tracker.positions[-1]
            x, y = int(last_pos[0]), int(last_pos[1])
            
            # Draw a filled dot for the vehicle center
            cv2.circle(frame, (x, y), 8, tracker.color, -1)
            
            # Only show speed if it has been calculated
            if tracker.speed_mph > 5:
                speed_text = f"{tracker.speed_mph:.0f}"
                
                # Draw white text with a black outline for readability
                cv2.putText(frame, speed_text, 
                           (x + 15, y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4) # Black outline
                cv2.putText(frame, speed_text, 
                           (x + 15, y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # White text

def draw_debug_grid(frame):
    """
    Draws a 10x10 grid with fractional coordinate labels (0.0 to 1.0)
    to help with tuning zones.
    """
    height, width, _ = frame.shape
    
    # Draw vertical lines (X-axis)
    for i in range(1, 10): # 0.1 to 0.9
        x = int(i / 10.0 * width)
        frac_x = i / 10.0
        cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)
        cv2.putText(frame, f"{frac_x:.1f}", (x + 5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw horizontal lines (Y-axis)
    for i in range(1, 10): # 0.1 to 0.9
        y = int(i / 10.0 * height)
        frac_y = i / 10.0
        cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)
        cv2.putText(frame, f"{frac_y:.1f}", (5, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_debug_zones(frame, start_zone_px, end_zone_px):
    """
    Draws the outlines of the detection zones for tuning.
    START_ZONE = Blue
    END_ZONE = Red
    """
    # Draw START zone (Blue)
    cv2.polylines(frame, [start_zone_px], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(frame, "START ZONE", (start_zone_px[0][0], start_zone_px[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw END zone (Red)
    cv2.polylines(frame, [end_zone_px], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, "END ZONE", (end_zone_px[0][0], end_zone_px[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)