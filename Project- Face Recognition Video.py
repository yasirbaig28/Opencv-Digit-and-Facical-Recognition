# FACIAL RECOGNITION FROM LIVE VIDEO
import cv2
import numpy as np

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video=cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face.detectMultiScale(gray,1.05,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    cv2.imshow('face',frame)

    if cv2.waitKey(1)==ord('q'):
        break
video.release()
cv2.destroyAllWindows()



