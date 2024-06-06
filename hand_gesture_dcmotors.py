# -- coding: utf-8 --
"""
Created on Sat Apr 20 12:40:45 2024

@author: aniru
"""

import cv2
import serial
bt=serial.Serial('COM19',9600)

from cvzone.HandTrackingModule import HandDetector
cap=cv2.VideoCapture(0)
detector=HandDetector(detectionCon=0.5,maxHands=1)
while True:
    ret,frame=cap.read()
    #frame=cv2.flip(frame,-1)
    hands,frame=detector.findHands(frame)
    if not hands:
        print("nothing")
    else:
        hands1=hands[0]
        fingers=detector.fingersUp(hands1) 
        #print(fingers) 
        count=fingers.count(1)
        print(count)
        string='X{0}'.format(count)
        bt.write(string.encode("utf-8"))
    cv2.imshow("FRAME", frame) 
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()