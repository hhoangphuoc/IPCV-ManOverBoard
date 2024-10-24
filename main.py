import cv2 as cv
import numpy as np

cap=cv.VideoCapture('input video.mp4')

#number of frames
n_frame=int(cap.get(cv.CAP_PROP_FRAME_COUNT))

#width and height of frame
w=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv.CAP_PROP_FPS))
#Define the codec of output video
fourcc=cv.VideoWriter_fourcc(*'MJPG')

#Setup output video
out=cv.VideoWriter('Video output.mp4', fourcc,fps, (w,h))

_, prev=cap.read()

prev_grey=cv.cvtColor(prev,cv.COLOR_BGR2GRAY)