import cv2
import numpy as np
from typing import Tuple

def estimate_wave_motion(frame1: np.ndarray, frame2: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Estimate uniform optical flow of sea waves within ROI using Lucas-Kanade method
    """
    x, y, w, h = roi
    roi1 = frame1[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)]
    roi2 = frame2[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)]
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    
    # Detect features to track
    features = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7)
    
    if features is not None:
        # Calculate optical flow
        flow, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, features, None)
        
        # Filter out points where flow wasn't found
        good_features = features[status == 1]
        good_flow = flow[status == 1]
        
        if len(good_flow) > 0:
            # Calculate median flow to get uniform motion
            flow_vectors = good_flow - good_features
            median_flow = np.median(flow_vectors, axis=0)
            print(f"Flow vector: {median_flow}")
            return median_flow #the flow vector
            
    return np.array([0, 0])

def compensate_wave_motion(frame: np.ndarray, flow_vector: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Apply motion compensation to nullify wave motion
    """
    x, y, w, h = roi
    roi_frame = frame[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)]
    
    # Create transformation matrix for motion compensation
    M = np.float32([[1, 0, -flow_vector[0]], 
                    [0, 1, -flow_vector[1]]])
    
    # Apply motion compensation
    compensated_roi = cv2.warpAffine(roi_frame, M, (w, h))
    return compensated_roi

def predict_buoy_position(prev_positions: list[Tuple[int, int]], frame: np.ndarray, dt: float = 1.0) -> Tuple[int, int]:
    """
    Predict the buoy's position using a simple motion model.
    """
    # Check if there are enough previous positions to calculate velocity
    if len(prev_positions) < 2:
        print("Not enough position history for motion model prediction")
        return None
    
    last_pos = prev_positions[-1] #current position of the buoy
    prev_pos = prev_positions[-2] #previous position of the buoy
    
    velocity_x = last_pos[0] - prev_pos[0]
    velocity_y = last_pos[1] - prev_pos[1]
    
    predicted_x = last_pos[0] + velocity_x
    predicted_y = last_pos[1] + velocity_y

    return int(predicted_x), int(predicted_y)

def predict_buoy_position_kalman(
        prev_positions: list[Tuple[int, int]],
        dt: float = 1.0) -> Tuple[int, int]:
    """
    Predict the buoy's position using a Kalman Filter.
    
    Args:
        prev_positions: List of previous buoy positions (x,y)
        frame: Current video frame
        dt: Time step between frames
        
    Returns:
        Predicted (x,y) position of buoy
    """
    if len(prev_positions) < 2:
        print("Not enough position history for Kalman prediction")
        return None
        
    # Initialize Kalman filter
    kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x,y,vx,vy), 2 measurements (x,y)
    
    # State transition matrix
    kalman.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # Initialize state with last known position and velocity
    last_pos = prev_positions[-1]
    prev_pos = prev_positions[-2]
    vx = (last_pos[0] - prev_pos[0]) / dt
    vy = (last_pos[1] - prev_pos[1]) / dt
    
    kalman.statePre = np.array([[last_pos[0]], [last_pos[1]], [vx], [vy]], dtype=np.float32)
    
    # Measurement matrix (we only measure position)
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    
    # Process noise covariance
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    
    # Measurement noise covariance
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    
    # Predict next state
    prediction = kalman.predict()
    
    predicted_x = int(prediction[0][0])
    predicted_y = int(prediction[1][0])
    
    return predicted_x, predicted_y

