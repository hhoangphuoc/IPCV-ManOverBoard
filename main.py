import cv2 as cv
import numpy as np
SMOOTHING_RADIUS = 30  # Reduced smoothing radius

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame, zoom_factor=1.1):  # Increase zoom factor to 1.1 (10% zoom)
    s = frame.shape
    T = cv.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, zoom_factor)
    frame = cv.warpAffine(frame, T, (s[1], s[0]))
    return frame

cap = cv.VideoCapture('input video.mp4')
n_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Change codec to XVID
out = cv.VideoWriter('Video output.mp4', fourcc, fps, (w, h))

_, prev = cap.read()
prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
transforms = np.zeros((n_frame - 1, 3), np.float32)


for i in range(n_frame - 2):
    prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=30)
    success, curr = cap.read()
    if not success:
        break
    curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
    curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    assert prev_pts.shape == curr_pts.shape
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    m, _ = cv.estimateAffinePartial2D(prev_pts, curr_pts)

    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])
    transforms[i] = [dx, dy, da]
    prev_gray = curr_gray
    print("Frame:", i, "/", n_frame, " - Tracker Points:", len(prev_pts))

trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth(trajectory)
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference * 0.5  # Reduce adjustments

cap.set(cv.CAP_PROP_POS_FRAMES, 0)

for i in range(n_frame - 2):
    success, frame = cap.read()
    if not success:
        break
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    frame_stabilized = cv.warpAffine(frame, m, (w, h))
    frame_stabilized = fixBorder(frame_stabilized, zoom_factor=1.1)  # Apply 10% zoom to reduce black borders
    frame_out = cv.hconcat([frame, frame_stabilized])

    if frame_out.shape[1] > 1920:
        frame_out = cv.resize(frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2))

    cv.imshow("Before and After", frame_out)
    cv.waitKey(10)
    out.write(frame_stabilized)

cap.release()
out.release()
cv.destroyAllWindows()


