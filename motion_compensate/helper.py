
import cv2
import numpy as np
from typing import Tuple, Optional, List

INTRINSIC_MATRIX = None
DISTORTION_COEFFS = None

def motion_compensate_and_subtract(curr_frame, prev_frame, pb_pred, optical_flow):
    """
    Compensates for motion between frames and subtracts background changes.
    Args:
        curr_frame: The current frame (in ROI region).
        prev_frame: The previous frame. (in ROI region).
        pb_pred: The predicted buoy position.
        optical_flow: The optical flow between the frames.
        l: The half-length of the buoy region.
        L: The half-width of the buoy region.
    Returns:
        The predicted buoy position and the substracted image.
    """
    prev_roi = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_roi = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Estimate flow between previous and current ROI
    if optical_flow is None:
        optical_flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Compensate the motion based on the flow
    avg_flow_x = np.mean(optical_flow[..., 0])
    avg_flow_y = np.mean(optical_flow[..., 1])
    
    #Subtract the previous ROI from the current ROI
    diff = cv2.absdiff(curr_roi, prev_roi)
    pb_new_pred = pb_pred + np.round([avg_flow_y, avg_flow_x]).astype(int)

    return pb_new_pred, diff

def update_motion_model(
          new_position: Optional[Tuple[int, int]],
          current_position: Tuple[int, int],
          dt: float = 1.0,
            max_frames_lost: int = 10,
            acceleration: Optional[Tuple[float, float]] = None,
            velocity: Optional[Tuple[float, float]] = None
          
    ):
        """
        Update the motion model based on new detection
        The new position is calcualted by the changes in velocity and acceleration
        
        Args:
            new_position: New detected position or None if not detected
            dt: Time step
        """
        # IF FOUND A NEW POSITION THEN ONLY UPDATE THE VELOCITY AND POSITION
        if new_position is not None:
            # Update velocity
            new_position = np.array(new_position, dtype=np.float32)
            # new_velocity = np.array(new_position) - np.array(self.current_position)
            new_velocity = (new_position - current_position) / dt
            acceleration = (new_velocity - velocity) / dt
            velocity = new_velocity

            # Update position
            prev_position = current_position
            current_position = tuple(map(int, new_position))
            frames_since_detection = 0

        else:
            # TODO: Predict position using motion model
            frames_since_detection += 1
            if frames_since_detection < max_frames_lost:
                velocity += acceleration * dt
                predicted_position = np.array(current_position) + velocity * dt
                current_position = tuple(map(int, predicted_position))

        return current_position, velocity, acceleration

def detect_horizon(intrinsic_matrix, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Detect horizon line using Hough transform and choosing the best line 
    then calculate rotation angles that require to rotate in order to stabilize the frame
    These are including: pitch angle (alpha) and tilt angle (beta)
    
    Args:
        frame: Input frame
        
    Returns:
        transformation_matrix: transformation matrix
    """
    # PARAMETERS
    horizon_threshold = 100
    min_theta = np.pi/3
    max_theta = 2*np.pi/3

    # Camera parameters
    camera_matrix = intrinsic_matrix #FIXME: Replace by the actual intrinsic matrix
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    fx = camera_matrix[0, 0] # focal length
    fy = camera_matrix[1, 1]

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 
                            threshold=horizon_threshold,
                            min_theta=min_theta,
                            max_theta=max_theta)
    
    if lines is None:
        return np.eye(3), 0, 0
        
    # TODO: FIND THE MOST HORIZONTAL LINE
    # best line that detected by Hough Transform, this result in rho and theta
    best_line = None
    min_angle_diff = float('inf')
    for rho, theta in lines[:, 0]:
        angle_diff = abs(abs(theta) - np.pi/2)
        if angle_diff < min_angle_diff:
            min_angle_diff = angle_diff
            best_line = (rho, theta)
            
    rho, theta = best_line
    
    # Calculate points of the line
    # x = rho * cos(theta) and y = rho * sin(theta)
    x0 = np.cos(theta) * rho
    y0 = np.sin(theta) * rho

    x1 = int(x0 + 1000*(-np.sin(theta)))
    y1 = int(y0 + 1000*(np.cos(theta)))
    x2 = int(x0 - 1000*(-np.sin(theta)))
    y2 = int(y0 - 1000*(np.cos(theta)))
    
    # Calculate rotation angles
    dy = cy - (y1 + (y2-y1)/(x2-x1)*(cx-x1))
    # pitch angle
    alpha = np.arctan(dy/fy)
    
    # rotation matrices for pitch angle
    R_alpha = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    
    # pitch correction
    M = camera_matrix @ R_alpha.T @ np.linalg.inv(camera_matrix)
    new_points = (M @ np.array([[x1, x2], [y1, y2], [1, 1]]))
    new_points = new_points[:2] / new_points[2]
    new_dir = new_points[:, 1] - new_points[:, 0]

    # calculate tilt angle
    beta = np.arctan2(new_dir[1], new_dir[0])
    # rotation matrices for tilt angle
    R_beta = np.array([
        [np.cos(beta), -np.sin(beta), 0],
        [np.sin(beta), np.cos(beta), 0],
        [0, 0, 1]
    ])
    
    # Final transformation matrix
    transform_matrix = camera_matrix @ R_beta.T @ R_alpha.T @ np.linalg.inv(camera_matrix)
    
    return transform_matrix #FIXME: Do we need to return?