import cv2
import numpy as np
import json
from opencv_process_video import overlay_zoomed_roi_debug, detect_horizon, estimate_distance
from evaluation.evaluate_params import AVERAGE_DISTANCE_ERROR, SUCCESS_RATE
def evaluate_tracking(
        video_path, 
        ground_truth_path, 
        threshold=5 #5 pixels
    ):
    """
    Evaluates the buoy tracking performance by comparing the tracked positions with ground truth.

    Args:
        video_path (str): Path to the video file.
        ground_truth_path (str): Path to the ground truth JSON file.
        threshold (int): Distance threshold (in pixels) for determining successful tracking.

    Returns:
        tuple: Average center distance error, tracking success rate.
    """

    cap = cv2.VideoCapture(video_path)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output video as .mp4
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

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

    # Specify the initial ROI (x, y, w, h)
    roi = (655, 532, 0, 0)

    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    center_distances = []
    success_count = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if str(frame_idx) in ground_truth:
            # Find Object and horizon
            frame, roi = overlay_zoomed_roi_debug(frame, roi, (50, 30), overlay_size=(200, 200), margin=10)
            start_coord, end_coord = detect_horizon(frame, roi)
            # print(f"start={start_coord}, end={end_coord}")

            # Estimate distance
            horizon = (start_coord[1] + end_coord[1]) / 2
            y0 = abs(horizon - roi[1])
            frame, dist = estimate_distance(frame, y0)
            # print(f"Distance to buoy:{dist}")

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)

            gt_x, gt_y = ground_truth[str(frame_idx)]
            tracked_x, tracked_y, _, _ = roi 

            distance = np.sqrt((gt_x - tracked_x)**2 + (gt_y - tracked_y)**2)
            center_distances.append(distance)

            if distance <= threshold:
                success_count += 1

            total_frames += 1

            key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed

            # Check if the 'N' key is pressed to move to the next frame
            if key == ord('n'):
                continue  # Proceed to the next frame in the loop

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            continue

    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()

    avg_distance_error = np.mean(center_distances)
    success_rate = (success_count / total_frames) * 100

    return avg_distance_error, success_rate


if __name__ == '__main__':
    video_path = 'input/stabilized.mp4'  # Replace with your video file path
    ground_truth_path = 'evaluate/ground_truth.json'  # Replace with your ground truth file path

    avg_error, success_rate = evaluate_tracking(video_path, ground_truth_path)

    print(f"Average Center Distance Error: {avg_error:.2f} pixels")
    print(f"Tracking Success Rate: {success_rate:.2f}%")

    # UPDATE THE GLOBAL VARIABLES
    AVERAGE_DISTANCE_ERROR = avg_error
    SUCCESS_RATE = success_rate

    print(f"Average Center Distance Error: {AVERAGE_DISTANCE_ERROR:.2f} pixels")
    print(f"Tracking Success Rate: {SUCCESS_RATE:.2f}%")
