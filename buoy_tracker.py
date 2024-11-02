import opencv_process_video

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #input_video = "input/stabilized_video.MP4"   # Make sure the input video file is in the correct location
    input_video = "input/stabilized_video.MP4"  # Make sure the input video file is in the correct location
    output_video = "output/buoy_tracker_motion.mp4"  # The processed video will be saved as this file
    opencv_process_video.main(input_video, output_video)

