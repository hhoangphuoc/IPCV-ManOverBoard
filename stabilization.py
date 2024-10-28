import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("input video.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the original video properties for consistent output
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Params for feature detection and tracking
feature_params = dict(maxCorners=200, qualityLevel=0.05, minDistance=30, blockSize=3)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Transformation accumulator for stabilization
transforms = []

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    # Keep only good points
    good_prev_pts = prev_pts[status == 1]
    good_curr_pts = curr_pts[status == 1]

    # Estimate the transformation
    m, _ = cv2.estimateAffinePartial2D(good_prev_pts, good_curr_pts)
    dx, dy = m[0, 2], m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])

    # Store transformations
    transforms.append([dx, dy, da])

    prev_gray = curr_gray.copy()
    prev_pts = good_curr_pts.reshape(-1, 1, 2)

# Smooth transformations
def smooth_trajectory(transforms, smoothing_radius=40):
    smoothed_transforms = []
    for i in range(len(transforms)):
        start = max(0, i - smoothing_radius)
        end = min(len(transforms) - 1, i + smoothing_radius)

        avg_dx = np.mean([t[0] for t in transforms[start:end + 1]])
        avg_dy = np.mean([t[1] for t in transforms[start:end + 1]])
        avg_da = np.mean([t[2] for t in transforms[start:end + 1]])

        smoothed_transforms.append([avg_dx, avg_dy, avg_da])

    return smoothed_transforms

smoothed_transforms = smooth_trajectory(transforms)

# Reset the capture to the start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Apply the smoothed transforms and write to output
for i, (dx, dy, da) in enumerate(smoothed_transforms):
    ret, frame = cap.read()
    if not ret:
        break

    # Create the transformation matrix with smoothed values
    transform = np.array([[np.cos(da), -np.sin(da), dx],
                          [np.sin(da), np.cos(da), dy]])

    # Apply the transform to stabilize the frame
    stabilized_frame = cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))

    # Write the stabilized frame to output video
    out.write(stabilized_frame)

    # Optional: Show the stabilized frame
    cv2.imshow("Stabilized Frame", stabilized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
