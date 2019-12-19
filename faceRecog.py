import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w] #(ycord_start,ycord_end),(xcod_start,xcord_end)
        img_item='my_img.png'

        id_,conf =recognizer.predict(roi_gray)
        if conf>45 and conf<85:
            print (id_)
        roi_color=frame[y:y+h,x:x+w]
        img_item2='my_color_img.png'
        cv2.imwrite(img_item,roi_gray)
        cv2.imwrite(img_item2,roi_color)

        color=(255,0,0)#BGR
        stroke=2
        width=x+w
        height=y+h
        cv2.rectangle(gray,(x,y),(width,height),color,stroke)

    cv2.imshow('frame1',gray)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break