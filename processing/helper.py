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

# FIXME: TRY WITH KALMAN FILTER - NOT WORKING --------------------------------
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
#-----------------------------------------------------------------------------
