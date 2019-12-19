import numpy as np
import cv2

cap =cv2.VideoCapture(0) #for selecting default webcam

filename='video111.avi'
frames_per_second=24.0



def rescale_frame(frame,percent=75):
    scale_percent=75
    width=int(frame.shape[1]*scale_percent/100)
    height=int(frame.shape[0]*scale_percent/100)
    dim=(width,height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


while True:
    ret, frame=cap.read()
    frame=rescale_frame(frame,percent=30)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow("Frame",frame) #not imgshow
    cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break
