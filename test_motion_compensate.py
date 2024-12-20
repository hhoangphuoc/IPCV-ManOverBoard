"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys
import numpy as np
from processing.motion_model import estimate_wave_motion, compensate_wave_motion, predict_buoy_position, predict_buoy_position_kalman

# Helper function to determine if the current frame's time is within a specified interval
def between(cap, lower: int, upper: int) -> bool:
    """
    Checks if the current video frame's timestamp is within a specified time interval.

    Parameters:
        cap (cv2.VideoCapture): The video capture object.
        lower (int): The lower bound of the interval in milliseconds.
        upper (int): The upper bound of the interval in milliseconds.

    Returns:
        bool: True if the current frame's timestamp is within the interval, False otherwise.
    """
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

# Function to apply Sobel operator
def apply_sobel(frame):
    """
    Applies the Sobel operator to detect edges in the frame.

    Parameters:
        frame (np.array): The input frame to process.

    Returns:
        np.array: The frame with Sobel edge detection applied.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    sobel_combined = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
    return sobel_combined

def apply_subtitle(frame, subtitle):
    """
    Adds a subtitle text to the bottom of the frame.

    Parameters:
        frame (np.array): The frame to add the subtitle to.
        subtitle (str): The subtitle text to overlay on the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (50, frame.shape[0] - 50)
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    line_type = cv2.LINE_AA
    cv2.putText(frame, subtitle, position, font, font_scale, font_color, thickness, line_type)


def detect_horizon(frame, roi):
    """
    Detect horizon line using Hough transform by selecting the most horizontal line.
    Returns default values if no line is detected.
    """
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Extract ROI coordinates
    x, y = roi[0], roi[1]  # Get center coordinates
    w, h = 400, 50  # Fixed width and height for horizon detection
    
    # Calculate ROI boundaries with bounds checking
    roi_y_start = max(0, y - h)
    roi_y_end = min(frame_height, y)
    roi_x_start = max(0, x - int(w/2))
    roi_x_end = min(frame_width, x + int(w/2))
    
    # # Check if ROI is valid
    # if roi_y_end <= roi_y_start or roi_x_end <= roi_x_start:
    #     # Return default values if ROI is invalid
    #     return (0, int(h/2)), (w, int(h/2))
    
    # Extract the ROI from the frame
    roi_frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # Check if ROI is empty
    if roi_frame.size == 0:
        return (0, int(h/2)), (w, int(h/2))
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 150, 200)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80, minLineLength=100, maxLineGap=300)

    # Initialize variables to store the best (most horizontal) line
    best_line = None
    min_angle_diff = float('inf')

    # Define target angle (horizontal) in radians
    target_angle = 0  # Horizontal line has angle of 0 degrees

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle of the line
            angle = np.arctan2(y2 - y1, x2 - x1)

            # Find angle difference from horizontal
            angle_diff = abs(angle - target_angle)
            # Update best line if this one is closer to horizontal
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_line = (x1, y1, x2, y2)
                
        if best_line is not None:
            print("Best line =", best_line)
            # Return the coordinates of the best line
            start = (best_line[0], best_line[1])
            end = (best_line[2], best_line[3])
            cv2.line(roi_frame, start, end, color=(0,0,255), thickness=2)
            return start, end

    # Return default values if no line is detected
    actual_width = roi_x_end - roi_x_start
    actual_height = roi_y_end - roi_y_start
    default_start = (0, int(actual_height/2))
    default_end = (actual_width, int(actual_height/2))
    cv2.line(roi_frame, default_start, default_end, color=(0,255,0), thickness=2)
    return default_start, default_end

def overlay_zoomed_roi_debug(frame, prev_frame, roi, size, overlay_size=(200, 200), margin=10, threshold_value=170, min_circularity=0.78, prev_positions=None):
    """
    Extracts, resizes, and overlays the zoomed ROI on the bottom right corner of the frame,
    applies thresholding, and identifies the largest circular white object by its center of mass and size.

    Parameters:
        frame (np.array): The input frame to process.
        prev_frame (np.array): Previous frame for motion model prediction.
        roi (tuple): A tuple (x, y, w, h) defining the region of interest within the frame.
        size (tuple): Width and height of the ROI to scan.
        overlay_size (tuple): Tuple (width, height) for the fixed overlay size.
        margin (int): Distance from the frame's edge for the overlay placement.
        threshold_value (int): Intensity threshold value for identifying white areas.
        min_circularity (float): Minimum circularity value to be considered circular.

        prev_positions (list): List of previous positions for predicting buoy position when it is not visible.

    Returns:
        frame (np.array): The frame with the overlay applied.
        center_of_mass (tuple): Coordinates (x, y) of the largest circular white blob's center of mass in the overlay.
    """
    x, y, w, h = roi
    w, h = size
    
    # TODO: ADDED WITH MOTION MODEL FOR PREV IF EXISTS --------------------------------
    # Estimate wave motion
    if prev_frame is not None:
        print("Compensating for wave motion...")
        wave_motion = estimate_wave_motion(prev_frame, frame, roi)
        # Compensate for wave motion
        roi_frame = compensate_wave_motion(frame, wave_motion, roi)
    #-----------------------------------------------------------------------------
    else:
        # Extract the ROI from the frame
        roi_frame = frame[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)]
    
    print(f"ROI (Original coordinates): {x, y}")

    # Resize the ROI to the fixed overlay size. For a 50x50 window, scaling would be 4x to accommodate 200x200
    zoomed_roi = cv2.resize(roi_frame, overlay_size, interpolation=cv2.INTER_CUBIC)

    # Calculate overlay position (bottom-right corner with margin)
    frame_height, frame_width = frame.shape[:2]
    roi_x_start = frame_width - overlay_size[0] - margin
    roi_y_start = frame_height - overlay_size[1] - margin

    # Convert the zoomed ROI to grayscale and apply threshold
    gray_frame = cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2GRAY)
    _, thresholded_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert single-channel thresholded image to 3-channel to match frame's color space
    thresholded_frame_bgr = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for tracking the largest circular white blob
    largest_area = 0
    center_of_mass = None

    # Loop through each contour to find the largest circular white blob
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            # Calculate perimeter and circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                print("Circularity:", circularity)

                # Check if the contour is circular enough
                if circularity >= min_circularity:
                    largest_area = area
                    # Calculate the center of mass for the largest circular blob
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        center_of_mass = (cX, cY)
                        print("COM on zoom =", center_of_mass)
                else:
                    print("No fit circle found 1")

    # Overlay the zoomed ROI
    'Change "zoomed_roi" to "thresholded_frame_bgr" for developing purposes'
    frame[roi_y_start:roi_y_start + overlay_size[1], roi_x_start:roi_x_start + overlay_size[0]] = zoomed_roi
    #frame[roi_y_start:roi_y_start + overlay_size[1], roi_x_start:roi_x_start + overlay_size[0]] = thresholded_frame_bgr

    # If a valid circular blob is found, update the ROI to center around the blob's COM
    if center_of_mass and 40 <= largest_area <= 260:
        # Calculate scale factors from overlay to original ROI size
        scale_x = w / overlay_size[0]
        scale_y = h / overlay_size[1]
        print("Scale =", scale_x, scale_y)

        # Transform the COM from overlay to the original frame coordinates
        original_cX = int(center_of_mass[0] * scale_x) + x - int(w/2)
        original_cY = int(center_of_mass[1] * scale_y) + y - int(h/2)
        
        # Update position history when buoy is visible
        if prev_positions is None:
            prev_positions = []
        prev_positions.append((original_cX, original_cY))
        if len(prev_positions) > 5:  # Keep last 5 positions
            prev_positions.pop(0)
            
        center_of_mass = (original_cX, original_cY, w, h)
        
        cv2.circle(frame, (original_cX, original_cY), 2, (0, 0, 255), 3)
        cv2.rectangle(frame, (original_cX-int(w/2), original_cY-int(h/2)),
                     (original_cX+int(w/2), original_cY+int(h/2)), color=(0, 0, 100))
        apply_subtitle(frame, "Tracking!")

    else:
        # Motion model prediction when buoy is not visible
        if prev_positions and len(prev_positions) >= 2:
            # predicted_x, predicted_y = predict_buoy_position(prev_positions, frame)
            predicted_x, predicted_y = predict_buoy_position_kalman(prev_positions) #FIXME: Replace with normal if not working
            
            # Apply wave motion compensation to predicted position
            if prev_frame is not None:
                wave_motion = estimate_wave_motion(prev_frame, frame, roi)
                predicted_x += wave_motion[0]
                predicted_y += wave_motion[1]
            
            # Boundary checking
            frame_height, frame_width = frame.shape[:2]
            predicted_x = max(int(w/2), min(frame_width - int(w/2), predicted_x))
            predicted_y = max(int(h/2), min(frame_height - int(h/2), predicted_y))
            
            # Update ROI to predicted position
            x, y = predicted_x, predicted_y
            
            # Draw predicted position
            cv2.circle(frame, (predicted_x, predicted_y), 2, (0, 255, 255), 3)  # Yellow dot for prediction
            cv2.rectangle(frame, (predicted_x-int(w/2), predicted_y-int(h/2)),
                         (predicted_x+int(w/2), predicted_y+int(h/2)), color=(0, 255, 100))
            
            center_of_mass = (predicted_x, predicted_y, w, h)

            apply_subtitle(frame, "Buoy Hidden! Predicting Position...")
        else:
            # Fallback when no previous positions are available
            center_of_mass = (x, y, w, h)
            cv2.rectangle(frame, (x-int(w/2), y-int(h/2)),
                         (x+int(w/2), y+int(h/2)), color=(0, 0, 100))
            apply_subtitle(frame, "Initializing Tracking...")

    # Print the center of mass and area of the largest blob
    if center_of_mass:
        print(f"Center of Mass: {center_of_mass}, Area: {largest_area}")
        print(f"Scan size: {w},{h}")

    return frame, center_of_mass, prev_positions
def main(input_video_file: str, output_video_file: str) -> None:
    # Interval for quick debug etc.
    interval = 1000
    offset = 250
    # Load templates for part 3f
    roi = cv2.imread('template/roi.png')

    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # Skip to the 120th frame
    frame_number = 0
    while frame_number < 120:
        ret, frame = cap.read()
        frame_number += 1
        if not ret:
            print("Could not read the 120th frame. Exiting.")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return

    # Allow the user to select the ROI on the 5th frame
    frame = apply_sobel(frame)
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    # while loop where the real work happens
    prev_frame = None
    prev_positions = None  # FIXME: This is used to track the buoy's position when it is not visible.
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            # Part 1
            if between(cap, 0, 10000):
                #track_with_csrt(cap, first_frame, roi)
                #frame = draw_white_circles_in_canny(frame, roi)
                #frame = threshold_white_values(frame, roi)
                pass

            frame, roi, prev_positions = overlay_zoomed_roi_debug(
                frame, prev_frame, roi, (50, 40), 
                overlay_size=(200, 200), margin=10,
                prev_positions=prev_positions  # Pass position history
            )

            start_coord, end_coord = detect_horizon(frame, roi)

            print(f"start={start_coord}, end={end_coord}")

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed

            prev_frame = frame.copy()

            # Check if the 'N' key is pressed to move to the next frame
            if key == ord('n'):
                continue  # Proceed to the next frame in the loop

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
