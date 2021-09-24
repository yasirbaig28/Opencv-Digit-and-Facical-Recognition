# FACIAL RECIGNITION for an Image
import cv2
import numpy as np

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#print(face)
image=cv2.imread('groupphoto.jpg')
#cv2.imshow('photo',image)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('photo',gray)

faces=face.detectMultiScale(gray,1.05,7)

for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

cv2.imshow('face',image)
