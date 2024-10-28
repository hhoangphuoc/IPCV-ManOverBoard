import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("input_video.mp4")

# Params for feature detection and tracking
feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
_, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Transform accumulator for stabilization
transforms = []

while True:
    # Read next frame
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    # Only keep good points
    good_prev_pts = prev_pts[status == 1]
    good_curr_pts = curr_pts[status == 1]

    # Estimate transformation
    m, _ = cv2.estimateAffinePartial2D(good_prev_pts, good_curr_pts)
    
    # Decompose transformation matrix into translation and rotation components
    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])
    
    # Accumulate transformations
    transforms.append([dx, dy, da])

    # Apply inverse transformation to stabilize the frame
    m_inv = cv2.invertAffineTransform(m)
    stabilized_frame = cv2.warpAffine(curr_frame, m_inv, (curr_frame.shape[1], curr_frame.shape[0]))

    # Display or save the stabilized frame
    cv2.imshow("Stabilized Frame", stabilized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update variables for next iteration
    prev_gray = curr_gray.copy()
    prev_pts = good_curr_pts.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
