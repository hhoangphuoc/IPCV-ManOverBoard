import cv2
import numpy as np

filename = "input video.MP4"
cap = cv2.VideoCapture(filename)

# Video writer for stabilized output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("stabilized.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (1280, 720))

# Parameters for template and search region
template_orig = np.array([1100, 800])  # [x, y] upper left corner
template_size = np.array([300, 50])    # [width, height]
search_border = np.array([15, 10])     # max horizontal and vertical displacement
template_center = (template_size - 1) // 2
template_center_pos = template_orig + template_center - 1
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Border for stabilization
BorderCols = np.r_[0:search_border[0] + 4, W - search_border[0] - 4:W]
BorderRows = np.r_[0:search_border[1] + 4, H - search_border[1] - 4:H]
sz = np.array([W, H])

# Initialize indices for template and search region
TargetRowIndices = np.arange(template_orig[1] - 1, template_orig[1] + template_size[1] - 1)
TargetColIndices = np.arange(template_orig[0] - 1, template_orig[0] + template_size[0] - 1)
SearchRegion = template_orig - search_border - 1
Offset = np.array([0, 0])
Target = np.zeros((18, 22, 3), dtype=np.uint8)
firstTime = True

# Update search function
def update_search(sz, motion_vector, search_region, offset, pos):
    A_i = offset - motion_vector
    AbsTemplate = pos['template_orig'] - A_i
    SearchTopLeft = AbsTemplate - pos['search_border']
    SearchBottomRight = SearchTopLeft + pos['template_size'] + 2 * pos['search_border']
    inbounds = np.all([(SearchTopLeft >= [0, 0]), (SearchBottomRight <= np.flip(sz))])

    if inbounds:
        mv_out = motion_vector
    else:
        mv_out = np.array([0, 0])

    offset -= mv_out
    search_region += mv_out
    return offset, search_region

# Read frames and process stabilization
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if firstTime:
        Idx = template_center_pos.copy()
        MotionVector = np.array([0, 0])
        firstTime = False
    else:
        IdxPrev = Idx.copy()
        x, y, w, h = *SearchRegion, *template_size + 2 * search_border
        ROI = frame[y:y+h, x:x+w]

        # Template Matching in color
        res = cv2.matchTemplate(ROI, Target, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        Idx = np.array(max_loc) + SearchRegion  # Adjust to full image coordinates
        MotionVector = Idx - IdxPrev

    # Update Offset and SearchRegion
    Offset, SearchRegion = update_search(sz, MotionVector, SearchRegion, Offset, {
        'template_orig': template_orig,
        'template_size': template_size,
        'search_border': search_border
    })

    # Translate frame for stabilization in color
    M = np.float32([[1, 0, -Offset[0]], [0, 1, -Offset[1]]])
    Stabilized = cv2.warpAffine(frame, M, (W, H))

    # Extract target for next frame's matching
    Target = Stabilized[TargetRowIndices[:, None], TargetColIndices]

    # Add black border
    Stabilized[:, BorderCols] = 0
    Stabilized[BorderRows, :] = 0

    # Draw rectangles on input to show target and search region
    TargetRect = np.hstack((template_orig - Offset, template_size))
    SearchRegionRect = np.hstack((SearchRegion, template_size + 2 * search_border))

    input_frame = frame.copy()
    input_frame = cv2.rectangle(input_frame, TargetRect[:2], TargetRect[:2] + TargetRect[2:], (255, 255, 255), 2)
    input_frame = cv2.rectangle(input_frame, SearchRegionRect[:2], SearchRegionRect[:2] + SearchRegionRect[2:], (255, 255, 255), 2)

    # Display Offset values
    cv2.putText(input_frame, f"({Offset[0]:+05.1f},{Offset[1]:+05.1f})", (191, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display video side by side: original and stabilized
    combined_frame = np.hstack((input_frame, Stabilized))
    resized_output = cv2.resize(combined_frame, (1280, 720))
    video=cv2.resize(Stabilized, (1280, 720))
    
    cv2.imshow("Video Stabilization", resized_output)
    
    # Write stabilized output frame
    out.write(video)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
