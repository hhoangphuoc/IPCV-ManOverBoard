"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys
import numpy as np

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)  # 4 dynamic parameters (x, y, dx, dy), 2 measurement parameters (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                   [0, 1, 0, 1],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]], np.float32) * 0.03

# Global variables to track previous positions
prev_positions = []
is_initialized = False

# Global variables to track previous frames
prev_frame = None
prev_gray = None

# Add these as global variables
last_valid_position = None
last_valid_frame = None

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


def detect_horizon(frame,roi):
    """
    Detect horizon line using Hough transform by selecting the most horizontal line.

    Parameters:
        frame (np.array): Input frame to process.

    Returns:
        start (tuple): Begin (x1, y1) of the detected horizon line.
        end (tuple: End coordinate (x2, y2) of the detected horizon
    """

    x, y, w, h = roi
    w=400
    h=50
    # Extract the ROI from the frame
    roi_frame = frame[y-h:y, x-int(w/2):x + int(w/2)]

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
    target_angle = 0  # Horizontal line has an angle of 0 degrees

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
                print(best_line)

    # Return the coordinates of the best line
    start = (best_line[0] + x - int(w / 2), best_line[1] + (y - h))
    end = (best_line[2] + x - int(w / 2), best_line[3] + (y - h))
    # start=0
    # end = 0
    cv2.line(frame, start, end, color=(0,0,255), thickness=2)
    return start, end

def estimate_distance(frame, y_0, focal_length = 1.67514300e+03, radius_earth = 6378100, camera_height = 2.5):
    # Distance to horizon
    D = np.sqrt((radius_earth+camera_height)**2 - radius_earth**2)

    # Slope of line
    m = y_0/focal_length

    # Resulting quadratic
    a = (1+m**2)
    b = -2*D - 2*m*radius_earth
    c = D**2

    # Find z coordinate
    z = (-b - np.sqrt(b**2-4*a*c))/(2*a)

    # Corresponding y coordinate
    y = m*z

    # Define vectors and calculate angle between them
    v1 = np.array([z-D,y-radius_earth])
    v2 = np.array([-D,-radius_earth])
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    dot_product = np.dot(v1,v2)

    # Calculate distance
    theta = np.arccos(dot_product/(v1_norm*v2_norm))
    distance = radius_earth*theta

    # Write to frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (int(frame.shape[1]/2)-200, frame.shape[0] - 50)
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    line_type = cv2.LINE_AA
    cv2.putText(frame, f"Distance to buoy: {distance:.2f} m", position, font, font_scale, font_color, thickness, line_type)
    return frame, distance

def compensate_motion(current_frame, prev_frame, roi):
    """
    Enhanced motion compensation focusing on ROI and handling different motion types
    """
    if prev_frame is None:
        return current_frame, None, None
    
    x, y, w, h = roi
    
    # Convert frames to grayscale
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # Calculate optical flow using Farneback method
    # flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
    #                                   pyr_scale=0.5, levels=3, winsize=15, 
    #                                   iterations=3, poly_n=5, poly_sigma=1.2, 
    #                                   flags=0)
    # Extract ROI for focused motion analysis
    roi_curr = curr_gray[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)]
    roi_prev = prev_gray[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)]
    # # Create motion compensation matrix
    # h, w = current_frame.shape[:2]
    # y_coords, x_coords = np.mgrid[0:h, 0:w].reshape(2, -1)
    # coords = np.vstack((x_coords, y_coords))
    # Detect key points in ROI
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(roi_prev, None)
    kp2, des2 = orb.detectAndCompute(roi_curr, None)
    
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return current_frame, None, None
    # # Apply flow to coordinates
    # flow_coords = coords + flow.reshape(2, -1)
    # Match key points
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Create transformation matrix
    # transformation_matrix = cv2.estimateAffinePartial2D(
    #     coords.T.reshape(-1, 1, 2),
    #     flow_coords.T.reshape(-1, 1, 2),
    #     method=cv2.RANSAC,
    #     ransacReprojThreshold=3,
    #     maxIters=2000
    # )[0]
    # Adjust points to global frame coordinates
    src_pts += np.float32([x-int(w/2), y-int(h/2)])
    dst_pts += np.float32([x-int(w/2), y-int(h/2)])
    
    # Calculate dominant motion vector in ROI
    if len(matches) >= 4:
        # Estimate rigid transformation (rotation + translation)
        transformation_matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, 
            ransacReprojThreshold=3, maxIters=2000
        )
        if transformation_matrix is not None:
            # Decompose transformation to get dominant vertical motion
            scale = np.sqrt(transformation_matrix[0,0]**2 + transformation_matrix[0,1]**2)
            angle = np.arctan2(transformation_matrix[0,1], transformation_matrix[0,0])
            translation = transformation_matrix[:, 2]
            # return compensated_frame, transformation_matrix
            # Create compensation matrix focusing on vertical motion
            compensation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), translation[0]],
                [np.sin(angle), np.cos(angle), translation[1]]
            ], dtype=np.float32)
            
            # Apply compensation
            h, w = current_frame.shape[:2]
            compensated_frame = cv2.warpAffine(current_frame, compensation_matrix, (w, h))
            
            # Calculate wave motion vector
            wave_motion = np.median(dst_pts - src_pts, axis=0).flatten()
            
            return compensated_frame, compensation_matrix, wave_motion
    
    return current_frame, None, None

def overlay_zoomed_roi_debug(frame, roi, size, overlay_size=(200, 200), margin=10, threshold_value=170, min_circularity=0.76):
    """
    Extracts, resizes, and overlays the zoomed ROI on the bottom right corner of the frame,
    applies thresholding, and identifies the largest circular white object by its center of mass and size.

    Parameters:
        frame (np.array): The input frame to process.
        roi (tuple): A tuple (x, y, w, h) defining the region of interest within the frame.
        size (tuple): Width and height of the ROI to scan.
        overlay_size (tuple): Tuple (width, height) for the fixed overlay size.
        margin (int): Distance from the frame's edge for the overlay placement.
        threshold_value (int): Intensity threshold value for identifying white areas.
        min_circularity (float): Minimum circularity value to be considered circular.

    Returns:
        frame (np.array): The frame with the overlay applied.
        center_of_mass (tuple): Coordinates (x, y) of the largest circular white blob's center of mass in the overlay.
    """
    #Global variables to apply Kalman Filter for
    # tracking the buoy based on previous positions
    global kalman, prev_positions, is_initialized, prev_frame, last_valid_position, last_valid_frame
    
    # Motion compensation with wave motion analysis
    compensated_frame, transform_matrix, wave_motion = compensate_motion(frame.copy(), prev_frame, roi)
    
    # Store current frame for next iteration
    prev_frame = frame.copy()
    
    # If we have wave motion information, adjust Kalman filter parameters
    if wave_motion is not None:
        # Adjust process noise based on wave motion magnitude
        wave_magnitude = np.linalg.norm(wave_motion)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], np.float32) * (0.03 * wave_magnitude)
        
        # Update transition matrix to favor vertical motion
        kalman.transitionMatrix = np.array([[1, 0, 0.1, 0],
                                          [0, 1, 0, 0.9],  # Increased weight for vertical motion
                                          [0, 0, 0.1, 0],
                                          [0, 0, 0, 0.9]], np.float32)
    
    # Adjust previous positions based on wave motion compensation
    if transform_matrix is not None and prev_positions:
        adjusted_positions = []
        for pos in prev_positions:
            pt = np.array([[pos[0], pos[1], 1]], dtype=np.float32).T
            transformed_pt = np.dot(transform_matrix, pt)
            adjusted_positions.append((int(transformed_pt[0]), int(transformed_pt[1])))
        prev_positions = adjusted_positions

    # Extract ROI from motion-compensated frame
    x, y, w, h = roi
    # w,h = size
    # print(f"ROI (Original coordinates): {x, y}")

    # # Extract the ROI from the frame
    # roi_frame = frame[y-int(h/2):y + int(h/2), x-int(w/2):x + int(w/2)]

    w, h = size
    # Extract the ROI from the compensated frame
    roi_frame = compensated_frame[y-int(h/2):y + int(h/2), x-int(w/2):x + int(w/2)]
    
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


    # If a valid circular blob is found, update tracking with compensated coordinates
    if center_of_mass and 25 <= largest_area <= 260:
        # Calculate scale factors from overlay to original ROI size
        scale_x = w / overlay_size[0]
        scale_y = h / overlay_size[1]
        print("Scale =", scale_x, scale_y)

        # Transform the COM from overlay to the original frame coordinates
        original_cX = int(center_of_mass[0] * scale_x) + x - int(w/2)
        original_cY = int(center_of_mass[1] * scale_y) + y - int(h/2)
        print("COM OG =", original_cX, original_cY)
        # center_of_mass = (original_cX, original_cY,w,h)
        
        # Store last valid position and frame for future reference
        last_valid_position = (original_cX, original_cY)
        last_valid_frame = frame.copy()
        
        # Update Kalman Filter with measurement
        measurement = np.array([[np.float32(original_cX)], [np.float32(original_cY)]])
        
        if not is_initialized:
            kalman.statePre = np.array([[np.float32(original_cX)], 
                                      [np.float32(original_cY)],
                                      [0], [0]], np.float32)
            kalman.statePost = kalman.statePre.copy()
            is_initialized = True
        
        kalman.correct(measurement)
        
        # Draw tracking visualization
        cv2.rectangle(frame, (original_cX-int(w/2), original_cY-int(h/2)),
                     (original_cX+int(w/2), original_cY+int(h/2)), color=(0, 0, 255))
        apply_subtitle(frame, "Buoy visible! Tracking...")

        center_of_mass = (original_cX, original_cY, w, h)
        
        #-------------------------------End of update -------------------------
        
    else:
        print("No valid blob found - Using last known position and motion compensation")
        if last_valid_position and last_valid_frame is not None:
            # Perform motion compensation between last valid frame and current frame
            compensated_frame, transform_matrix, wave_motion = compensate_motion(frame.copy(), last_valid_frame, roi)
            
            if transform_matrix is not None:
                # Transform last valid position using motion compensation
                last_x, last_y = last_valid_position
                pt = np.array([[last_x, last_y, 1]], dtype=np.float32).T
                transformed_pt = np.dot(transform_matrix, pt)
                
                predicted_x = int(transformed_pt[0])
                predicted_y = int(transformed_pt[1])
                
                # Apply vertical motion bias (buoys tend to move more vertically)
                if wave_motion is not None:
                    # Apply only a portion of the vertical wave motion
                    predicted_y += int(wave_motion[1] * 0.1)  # Reduced influence of wave motion
                    print("Predicted y =", predicted_y)

                    # Remove the horizontal wave motion
                    # predicted_x += int(wave_motion[0] * 0.0)
                
                # Bounds checking
                predicted_x = max(int(w/2), min(frame.shape[1]-int(w/2), predicted_x))
                predicted_y = max(int(h/2), min(frame.shape[0]-int(h/2), predicted_y))
                
                # Draw predicted position
                cv2.rectangle(frame, (predicted_x-int(w/2), predicted_y-int(h/2)),
                             (predicted_x+int(w/2), predicted_y+int(h/2)), color=(0, 255, 255))
                
                center_of_mass = (predicted_x, predicted_y, w, h)
                apply_subtitle(frame, "Predicting!")
            else:
                # If motion compensation fails, use last known position
                last_x, last_y = last_valid_position
                cv2.rectangle(frame, (last_x-int(w/2), last_y-int(h/2)),
                             (last_x+int(w/2), last_y+int(h/2)), color=(255, 165, 0))
                center_of_mass = (last_x, last_y, w, h)
                apply_subtitle(frame, "Buoy invisible! Predicting...")
        else:
            # Fallback to original search behavior
            w = 60
            h = 50
            center_of_mass = (x, y, w, h)
            cv2.rectangle(frame, (x-int(w/2), y-int(h/2)),
                         (x+int(w/2), y+int(h/2)), color=(0, 0, 100))
            apply_subtitle(frame, "Searching...")

    # Print the center of mass and area of the largest blob
    if center_of_mass:
        print(f"Center of Mass: {center_of_mass}, Area: {largest_area}")
        print(f"Scan size: {w},{h}")

    return frame, center_of_mass
def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # Skip to the 120th frame
    frame_number = 0
    while frame_number < 1:
        ret, frame = cap.read()
        frame_number += 1
        if not ret:
            print("Could not read the 120th frame. Exiting.")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return

    # Allow the user to select the ROI on the 5th frame
    #frame = apply_sobel(frame)
    #roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    #cv2.destroyWindow("Select ROI")
    roi = (655, 532,0,0)

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            # Part 1
            if between(cap, 0, 20000):
                #track_with_csrt(cap, first_frame, roi)
                #frame = draw_white_circles_in_canny(frame, roi)
                #frame = threshold_white_values(frame, roi)
                pass

            # Find Object and horizon
            frame, roi = overlay_zoomed_roi_debug(frame, roi, (50, 30), overlay_size=(200, 200), margin=10)
            start_coord, end_coord = detect_horizon(frame, roi)
            print(f"start={start_coord}, end={end_coord}")

            # Estimate distance
            horizon = (start_coord[1] + end_coord[1])/2
            y0 = abs(horizon - roi[1])
            frame, dist = estimate_distance(frame, y0)
            print(f"Distance to buoy:{dist}")

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed

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
