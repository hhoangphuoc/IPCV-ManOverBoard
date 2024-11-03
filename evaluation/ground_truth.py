import cv2
import json

def annotate_video(video_path, output_json_path):
    """
    Opens a video, allows the user to annotate the buoy position frame-by-frame,
    and saves the annotations to a JSON file.

    Args:
        video_path (str): Path to the video file.
        output_json_path (str): Path to save the ground truth JSON file.
    """

    cap = cv2.VideoCapture(video_path)
    ground_truth = {}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ground_truth[frame_idx] = [x, y]
            print(f"Annotated frame {frame_idx}: {x, y}")


    cv2.namedWindow('Annotation Window')
    cv2.setMouseCallback('Annotation Window', mouse_callback)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended or no frame read")
            break
        
        # # if q key is pressed, break the loop   
        # if cv2.waitKey(28) & 0xFF == ord('q'):
        #     break

        # while True:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if str(frame_idx) in ground_truth:
            x, y = ground_truth[str(frame_idx)]
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Annotation Window', frame)

        key = cv2.waitKey(0)

        # if enter key is pressed, break the loop
        if key == 13:
            continue
        elif key == ord('q'):
            break
        # # if n key is pressed, continue to the next frame
        # elif key == ord('n'):
        #     continue
    cap.release()
    cv2.destroyAllWindows()

    with open(output_json_path, 'w') as f:
        json.dump(ground_truth, f, indent=4)

    print(f"Ground truth saved to {output_json_path}")

if __name__ == '__main__':
    video_path = 'stabilized.mp4'  # Replace with your video file path
    output_json_path = 'ground_truth.json'  # Replace with desired output path
    annotate_video(video_path, output_json_path)