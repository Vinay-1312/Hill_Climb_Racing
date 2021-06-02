# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:24:47 2020

@author: DELL
"""
import time
import pyautogui as pg
import pydirectinput as pi
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
model=load_model("my_model2")
vid=cv2.VideoCapture(0)
hand=cv2.CascadeClassifier("Hand.Cascade.1.xml")
fist=cv2.CascadeClassifier("fist.xml")
for i in list(range(5))[::-1]:
    print(i+1)
    time.sleep(1)

i=0
while True:
    check,img=vid.read()
    img=cv2.flip(img,1,1) #Flip to act as a mirror
    #mini=cv2.resize(img,(img.shape[1]//4,img.shape[0]//4))
    #hands=hand.detectMultiScale(mini)
    #for h in hands:
        #(x,y,w,h)=[v*4 for v in h]
        ##hand_img=img[y:y+h,x:x+w]
    img1=cv2.resize(img,(150,150))
    normalized=img1/255

    img2=np.reshape(normalized,(1,150,150,3))
    reshape=np.vstack([img2])

    pred=model.predict(reshape)
        
    
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    
    p=np.argmax(pred)
    
    if p==0:
        if i==0:
            time.sleep(3)
        pi.keyUp("right")
        pi.keyDown("left")
        cv2.putText(img,  'Breaks',  
                (50, 50),  
                cv2.FONT_HERSHEY_SIMPLEX, 1,  
                (0, 255, 0),  
                2)
  

    if p==1:
        if i==0:
            time.sleep(3)
        pi.keyUp("left")
        pi.keyDown("right")
        cv2.putText(img,  'acc',  
                (50, 50),  
                cv2.FONT_HERSHEY_SIMPLEX, 1,  
                (0, 0, 255),2)
    cv2.imshow("game",img)
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()

