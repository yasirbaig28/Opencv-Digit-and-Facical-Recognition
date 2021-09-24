import cv2
import numpy as np
import sys

im=cv2.imread('train.png')

#cv2.imshow('image',im)
im3=im.copy()
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) 
blur=cv2.GaussianBlur(gray,(5,5),0)
#cv2.imshow('window',blur)
thresh= cv2.adaptiveThreshold(blur,255,1,1,11,2)
#cv2.imshow('Threshold',thresh)

contours, hierarchy=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
samples=np.empty((0,100),np.float32)
#print(samples)
responses=[]

keys=[i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h]=cv2.boundingRect(cnt)

        if h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi= thresh[y:y+h,x:x+w]
            roismall=cv2.resize(roi,(10,10))
            #print(roismall)
            #cv2.imshow('norm',roismall)
            cv2.imshow('norm',im)
            key=cv2.waitKey(0)

            if key==27:
                sys.exit()

            elif key in keys:
                responses.append(int(chr(key)))
                sample=roismall.reshape((1,100))
                samples=np.append(samples,sample,0)

responses= np.array(responses, np.float32)
responses= responses.reshape((responses.size,1))
print("training complete")

samples=np.float32(samples)
responses=np.float32(responses)

cv2.imwrite("train_result.png",im)
np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)  
 
