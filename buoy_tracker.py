import cv2
import numpy as np
from typing import Tuple, Optional, List
import math
import glob
from processing.helper import calibrate_camera, detect_horizon, stabilize_frame, detect_buoy
from processing.params import CAMERA_PARAMS

# Example usage
def main():
    input_video = "input/stabilized_video.MP4"
    output_video = "output/buoy_tracking_stabilized.MP4"


    # 1. CAMERA CALIBRATION --------------------------------
    # calibration_images = glob.glob("calibration images/*.jpg")
    # camera_params = calibrate_camera(calibration_images, board_size=(9, 6), square_size=40.0)

    # print("Camera parameters: ", camera_params)

    # CAMERA_PARAMS = camera_params
    #--------------------------------



    cap = cv2.VideoCapture(input_video)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output video as .mp4
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # 2. STABILIZATION --------------------------------
    prev_frame = None
    prev_transform = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        stabilized_frame = stabilize_frame(frame, prev_transform)
        prev_transform = stabilized_frame[1]
        out.write(stabilized_frame[0])
        prev_frame = stabilized_frame[0]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    #--------------------------------
    # close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()