import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict

from processing.params import CAMERA_PARAMS
from tqdm import tqdm

def calibrate_camera(calibration_images: List[str], board_size: Tuple[int, int] = (9, 6), square_size: float = 40.0) -> Dict:
    """
    Calibrate the camera using chessboard images
    
    Args:
        calibration_images: List of paths to calibration images
        board_size: Number of inner corners (width, height)
        square_size: Size of each square in mm
        
    Returns:
        Dict containing camera parameters (matrix, dist_coeffs, etc.)
    """
    # Prepare object points
    print("Start calibration...")
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    obj_points = []
    img_points = []
    
    for img_path in tqdm(calibration_images, desc="Processing calibration images..."):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners2)
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)
        
    if not ret:
        raise ValueError("Calibration failed")
        
    CAMERA_PARAMS['intrinsic_matrix'] = camera_matrix
    CAMERA_PARAMS['distortion_coeffs'] = dist_coeffs
    CAMERA_PARAMS['fx'] = camera_matrix[0, 0]
    CAMERA_PARAMS['fy'] = camera_matrix[1, 1]
    CAMERA_PARAMS['cx'] = camera_matrix[0, 2]
    CAMERA_PARAMS['cy'] = camera_matrix[1, 2]

    return CAMERA_PARAMS

def detect_horizon(frame: np.ndarray, frame_width: int, horizon_threshold: int = 150) -> Optional[np.ndarray]:
    """
    Detect horizon line using Hough transform and choosing the best line 
    then calculate rotation angles that require to rotate in order to stabilize the frame
    These are including: pitch angle (alpha) and tilt angle (beta)
    
    Args:
        frame: Input frame
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        transformation_matrix: transformation matrix
    """

    print("Detecting horizon...")

    # Convert to grayscale and detect edges
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, threshold1=0.2*255, threshold2=0.3*255, apertureSize=3)

    # More restrictive theta range for better horizontal line detection
    theta_range = np.concatenate((
        np.linspace(-95, -85, 50),  # Slightly wider range
        np.linspace(85, 95, 50)
    ))
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, 
                          threshold=horizon_threshold,
                          min_theta=np.deg2rad(min(theta_range)),
                          max_theta=np.deg2rad(max(theta_range)))
    
    if lines is None:
        return None

    # Find the most horizontal line
    best_line = None
    min_angle_diff = float('inf')
    
    for line in lines:
        rho, theta = line[0]
        # Convert theta to degrees for easier comparison
        angle = np.rad2deg(theta)
        # Find line closest to horizontal (90 or -90 degrees)
        angle_diff = min(abs(abs(angle) - 90), abs(abs(angle) + 90))
        if angle_diff < min_angle_diff:
            min_angle_diff = angle_diff
            best_line = line[0]

    if best_line is None or min_angle_diff > 15:  # Reject if too far from horizontal
        return None

    rho, theta = best_line
    
    # Calculate line points
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + frame_width*(-b))
    y1 = int(y0 + frame_width*(a))
    x2 = int(x0 - frame_width*(-b))
    y2 = int(y0 - frame_width*(a))

    return np.array([[x1, y1], [x2, y2]])

def calculate_transform_matrix(horizontal_points: np.ndarray) -> np.ndarray:
    """
    Calculate the transformation matrix
    based on the horizontal points found by Hough Transform
    """
    intrinsic_matrix = CAMERA_PARAMS['intrinsic_matrix']
    cx = CAMERA_PARAMS['cx']    
    cy = CAMERA_PARAMS['cy']
    fx = CAMERA_PARAMS['fx']
    fy = CAMERA_PARAMS['fy']

    p1, p2 = horizontal_points

    print("Using horizontal points to calculate transformation matrix...")
    print("p1 = ({}, {}), p2 = ({}, {})".format(p1[0], p1[1], p2[0], p2[1]))

    # Calculate pitch angle (alpha) with constraints
    dy = cy - (p1[1] + (p2[1]-p1[1])/(p2[0]-p1[0])*(cx-p1[0]))
    alpha = np.arctan(dy/fy)
    # Limit pitch angle to prevent extreme rotations
    alpha = np.clip(alpha, -np.pi/6, np.pi/6)  # Limit to ±30 degrees
    
    # Pitch rotation matrix
    R_alpha = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    
    # Calculate intermediate points after pitch correction
    M = intrinsic_matrix @ R_alpha.T @ np.linalg.inv(intrinsic_matrix)
    new_points = (M @ np.array([[p1[0], p2[0]], [p1[1], p2[1]], [1, 1]]))
    new_points = new_points[:2] / new_points[2]
    new_dir = new_points[:, 1] - new_points[:, 0]

    # calculate tilt angle
    beta = np.arctan2(new_dir[1], new_dir[0])
    # Limit tilt angle to prevent extreme rotations
    beta = np.clip(beta, -np.pi/6, np.pi/6)  # Limit to ±30 degrees
    
    # Tilt rotation matrix
    R_beta = np.array([
        [np.cos(beta), -np.sin(beta), 0],
        [np.sin(beta), np.cos(beta), 0],
        [0, 0, 1]
    ])
    
    # Final transformation matrix
    transform_matrix = intrinsic_matrix @ R_beta.T @ R_alpha.T @ np.linalg.inv(intrinsic_matrix)
    
    return transform_matrix

def stabilize_frame(frame: np.ndarray, prev_transform: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stabilize the frame using the transformation matrix
    created by Hough Transform
    """
    h, w = frame.shape[:2]

    horizontal_points = detect_horizon(frame, w)
    
    # Check if horizon detection failed
    if horizontal_points is None:
        return frame, np.eye(3)

    # Now we can safely unpack the points
    p1, p2 = horizontal_points
    
    # Calculate new transform
    transform_matrix = calculate_transform_matrix(horizontal_points)
    
    # Smooth transition with previous transform
    if prev_transform is not None:
        smoothing_factor = 0.8  # Adjust this value to control smoothing (0-1)
        transform_matrix = smoothing_factor * transform_matrix + (1 - smoothing_factor) * prev_transform

    # Get the corners of the frame
    corners = np.array([[0, 0, 1],
                       [w, 0, 1],
                       [w, h, 1],
                       [0, h, 1]]).T

    # Transform corners
    transformed_corners = transform_matrix @ corners
    transformed_corners = transformed_corners[:2] / transformed_corners[2]

    # Calculate scaling to maintain frame size
    src_corners = corners[:2].T
    dst_corners = transformed_corners.T
    
    # Calculate and apply scaling matrix to prevent frame shrinkage
    scale_x = w / (np.max(dst_corners[:, 0]) - np.min(dst_corners[:, 0]))
    scale_y = h / (np.max(dst_corners[:, 1]) - np.min(dst_corners[:, 1]))
    scale = min(scale_x, scale_y)
    
    scaling_matrix = np.array([
        [scale, 0, w*(1-scale)/2],
        [0, scale, h*(1-scale)/2],
        [0, 0, 1]
    ])

    # Combine transforms
    final_transform = scaling_matrix @ transform_matrix

    # Apply transformation with border handling
    stabilised_frame = cv2.warpPerspective(
        frame, 
        final_transform, 
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # draw the horizon line
    cv2.line(stabilised_frame, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 255), 2)

    return stabilised_frame, final_transform

def detect_buoy(frame: np.ndarray, current_position: Tuple[int, int], 
                roi_size: Tuple[int, int], template: Optional[np.ndarray] = None) -> Optional[Tuple[int, int]]:
    """
    Detect buoy position in the ROI 
    using template matching
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame")
        
    x, y = current_position
    w_center, h_center = roi_size[0] // 2, roi_size[1] // 2
    
    x1 = max(0, x-w_center)
    x2 = min(frame.shape[1], x+w_center)
    y1 = max(0, y-h_center)
    y2 = min(frame.shape[0], y+h_center)
    
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None
        
    if template is None:
        return current_position
        
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val > 0.7:
        new_x = x - w_center + max_loc[0]
        new_y = y - h_center + max_loc[1]
        return (new_x, new_y)
    
    return None

def estimate_buoy_distance(buoy_position: Tuple[int, int], 
                           intrinsic_matrix: np.ndarray,
                           camera_height: float = 2.5,
                           earth_radius: float = 6371000,
                           ) -> float:
    """
    Estimate the distance of the buoy.
    Args:
        buoy_position: Position of the buoy
        intrinsic_matrix: Camera matrix
    Returns:
        float: Distance to the buoy from the camera position
    """
    # camera center
    cx = intrinsic_matrix[0,2]
    cy = intrinsic_matrix[1,2]
    fx = intrinsic_matrix[0,0]
    fy = intrinsic_matrix[1,1]
    
    horizon_distance = np.sqrt((earth_radius + camera_height)**2 - earth_radius**2)  # horizon distance
    
    # distance in y
    y0 = buoy_position[0] - cy        # diff buoy horizon in the y axis
    gamma_y = np.arctan(y0/fy)  # angle y between z camera axis and buoy
    earth_angle = np.arctan(earth_radius/horizon_distance)  # angle between horizon line and vertical line of the camera
    distance_y = camera_height * np.tan(earth_angle - gamma_y)    # distance in y of the buoy
    
    # distance in x
    x0 = buoy_position[1] - cx    
    distance_x = x0 * distance_y/fx
    
    buoy_distance = np.sqrt(distance_x**2 + distance_y**2) # distance to the buoy = sqrt(x^2 + y^2)
    
    return buoy_distance

def create_kalman_filter():
    """Initialize OpenCV Kalman filter for tracking"""
    kf = cv2.KalmanFilter(4, 2)  # 4 dynamic parameters (x,y,dx,dy), 2 measurement parameters (x,y)
    
    # Measurement matrix (H)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0]], np.float32)
    
    # State transition matrix (F)
    kf.transitionMatrix = np.array([[1, 1, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 1]], np.float32)
    
    # Process noise covariance (Q)
    kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]], np.float32) * 0.03
    
    return kf

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
