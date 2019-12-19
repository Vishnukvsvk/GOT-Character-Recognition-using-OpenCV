import cv2
import numpy as np
import pickle


face_cascade=cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels_og = {'arya-stark': 0, 'bran-stark': 1, 'brienne-of-tarth': 2, 'bronn': 3, 'cersei-lannister': 4, 'davos': 5, 'eddard-stark': 6, 'ellaria-sand': 7, 'gendry': 8, 'gilly': 9, 'grey-worm': 10, 'hodor': 11, 'jaime-lannister': 12, 'jaqen-hghar': 13, 'joffrey-baratheon': 14, 'jojen-reed': 15, 'jon-snow': 16, 'jorah-mormont': 17, 'khal-drogo': 18, 'loras-tyrell': 19, 'lord-varys': 20, 'margaery-tyrell': 21, 'missandei': 22, 'myrcella-baratheon': 23, 'night-king': 24, 'oberryn-martell': 25, 'petyr-baelish': 26, 'podrick': 27, 'ramsay-bolton': 28, 'robb-stark': 29, 'roose-bolton': 30, 'samwell-tarly': 31, 'sandor-clegane': 32, 'sansa-stark': 33, 'shae': 34, 'stannis-baratheon': 35, 'theon-grejoy': 36, 'tormund': 37, 'tyrion-lannister': 38, 'walder-frey': 39, 'ygritte': 40}
labels={v: k for k, v in labels_og.items()}



ig=cv2.imread('kit.jpg')
gray=cv2.cvtColor(ig,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

for (x,y,w,h) in faces:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color = ig[y:y+h, x:x+w]
    id_,conf =recognizer.predict(roi_gray)
    if conf>20:    
        font=cv2.FONT_HERSHEY_SIMPLEX
        name=labels[id_]
        color=(34,139,34)
        stroke=2
        cv2.putText(ig,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

    img_item='recog_img.png'
    cv2.imwrite(img_item,roi_color)
    color=(255,0,0)#BGR
    stroke=2
    width=x+w
    height=y+h
    cv2.rectangle(ig,(x,y),(width,height),color,stroke)
cv2.imshow('frame',ig)
cv2.waitKey(0)
cv2.destroyAllWindows()
    


    


