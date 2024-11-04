import cv2
import numpy as np
import json
from opencv_process_video import overlay_zoomed_roi_debug, detect_horizon, estimate_distance
from evaluation.evaluate_params import AVERAGE_DISTANCE_ERROR, SUCCESS_RATE, AVERAGE_IOU, STD_DEV_DISTANCE, STD_DEV_IOU

import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IOU) 
    between two bounding boxes.

    Args:
        box1 (tuple): (x1, y1, w1, h1) of the first box.
        box2 (tuple): (x2, y2, w2, h2) of the second box.

    Returns:
        float: IOU value.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the (x, y) coordinates of the intersection rectangle
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # Calculate the area of intersection rectangle
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    # Calculate IOU
    iou = inter_area / float(union_area)

    return iou
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
    out = cv2.VideoWriter('evaluation/evaluate_output.mp4', fourcc, fps, (frame_width, frame_height))

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
    ious = [] #number of frames where iou is greater than 0.5

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

            gt_x, gt_y = ground_truth[str(frame_idx)]
            tracked_x, tracked_y, tracked_w, tracked_h = roi 

            # draw ground truth
            cv2.circle(frame, (gt_x, gt_y), 5, (0, 255, 0), 1) #green
            # draw tracked
            cv2.circle(frame, (tracked_x, tracked_y), 5, (0, 0, 255), 1) #red

            distance = np.sqrt(abs(gt_x - tracked_x)**2 + abs(gt_y - tracked_y)**2)
            # draw distance
            cv2.putText(frame, f"Distance: {distance:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            center_distances.append(distance)

            # calculate iou
            gt_rect = (int(gt_x - tracked_w/2), int(gt_y - tracked_h/2), tracked_w, tracked_h)  # Ground truth box
            print(f"Ground truth box: {gt_rect}")
            tracked_box = roi
            print(f"Tracked box: {tracked_box}")

            # draw gt box and tracked box
            cv2.rectangle(frame, gt_rect, (0, 255, 0), 2) #green

            gt_box = (gt_x, gt_y, tracked_w, tracked_h)
            iou = calculate_iou(gt_box, tracked_box) #calculate iou
            ious.append(iou)

            if distance <= threshold:
                success_count += 1

            total_frames += 1

            # draw iou
            cv2.putText(frame, f"IOU: {iou:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # draw success rate
            cv2.putText(frame, f"Success: {success_count}/{total_frames}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # draw frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
            continue

    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()

    # Calculate average distance error
    avg_distance_error = np.mean(center_distances)
    std_dev_distance = np.std(center_distances)

    # Calculate success rate
    success_rate = (success_count / total_frames) * 100

    # Calculate iou
    avg_iou = np.mean(ious)
    std_dev_iou = np.std(ious)

    return center_distances, ious,avg_distance_error, success_rate, avg_iou, std_dev_distance, std_dev_iou


if __name__ == '__main__':
    video_path = 'input/stabilized.mp4'  # Replace with your video file path
    ground_truth_path = 'evaluation/ground_truth.json'  # Replace with your ground truth file path

    center_distances, ious, avg_error, success_rate, avg_iou, std_dev_distance, std_dev_iou = evaluate_tracking(video_path, ground_truth_path)

    # print(f"Average Center Distance Error: {avg_error:.2f} pixels")
    # print(f"Tracking Success Rate: {success_rate:.2f}%")
    # print(f"Average IOU: {avg_iou:.2f}")
    # print(f"Standard Deviation of Center Distance: {std_dev_distance:.2f} pixels")
    # print(f"Standard Deviation of IOU: {std_dev_iou:.2f}")

    # plot the center distances, ious in one plot
    # Plot combined metrics
    plt.figure(figsize=(10, 5))
    plt.title('Combined Metrics: Distance Differences and IOU')
    plt.plot(center_distances,  label='Center Distances')
    plt.plot(ious, label='IOUs')
    plt.legend()
    plt.savefig('combined_metrics.png')
    plt.close()

    # Plot center distances separately
    plt.figure(figsize=(10, 5))
    plt.title('Center Distances Over Time')
    plt.plot(center_distances, 'orange', label='Center Distances')
    plt.ylabel('Distance (pixels)')
    plt.xlabel('Frame')
    plt.savefig('center_distances.png')
    plt.close()

    # Plot IOUs separately
    plt.figure(figsize=(10, 5))
    plt.title('Intersection Over Union (IOU) Over Time')
    plt.plot(ious, 'blue', label='IOU')
    plt.ylabel('IOU')
    plt.xlabel('Frame')
    plt.savefig('iou_metrics.png')
    plt.close()


    # UPDATE THE GLOBAL VARIABLES
    AVERAGE_DISTANCE_ERROR = avg_error
    SUCCESS_RATE = success_rate
    AVERAGE_IOU = avg_iou
    STD_DEV_DISTANCE = std_dev_distance
    STD_DEV_IOU = std_dev_iou

    print(f"Average Center Distance Error: {AVERAGE_DISTANCE_ERROR:.2f} pixels")
    print(f"Tracking Success Rate: {SUCCESS_RATE:.2f}%")
    print(f"Average IOU: {AVERAGE_IOU:.2f}")
    print(f"Standard Deviation of Center Distance: {STD_DEV_DISTANCE:.2f} pixels")
    print(f"Standard Deviation of IOU: {STD_DEV_IOU:.2f}")

