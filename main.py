# This is a sample Python script.
import opencv_process_video
#import stabilisation


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #input_video = "input/stabilized_video.MP4"   # Make sure the input video file is in the correct location
    input_video = "input/stabilized_video.MP4"  # Make sure the input video file is in the correct location
    output_video = "output/project_buoy_tracking.mp4"  # The processed video will be saved as this file
    opencv_process_video.main(input_video, output_video)
    #stabilisation()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
