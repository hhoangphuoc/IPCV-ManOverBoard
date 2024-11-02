
"""
THIS FUNCTION IS USING TO FIND THE INITIAL BUOY POSITION
AT THE TIMESTAMP OF 1 SECOND. THAT'S WHERE WE START THE TRACKING
"""
import cv2

INITIAL_BUOY_POSITION = (667, 540) # initial point position (width, height)
VIDEO_PATH = "input video.mp4" # Load an image or video frame

# Mouse callback function to capture the position of a click
def get_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left button click
        print(f"Point clicked: ({x}, {y})")


cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_time = 0 #start the tracking from second 1

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if cv2.waitKey(28) & 0xFF == ord('q'):
            break
    
    if cap.get(cv2.CAP_PROP_POS_MSEC) == start_time:
        first_frame = frame
        # Create a window and set a callback function for mouse events
        cv2.namedWindow("Initial Frame")
        cv2.setMouseCallback("Initial Frame", get_mouse_click)

        # Looping to display the image until 'q' is pressed
        while True:
            # Display the image
            cv2.imshow("Initial Frame", first_frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Close the window
        cv2.destroyAllWindows()
