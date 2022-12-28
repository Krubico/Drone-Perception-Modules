import cv2
import os
import numpy as np


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label, confidence = face_recognizer.predict(img)

    cv2.putText(img, str(people[label]) (20,20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), thickness=2)
    cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), thickness=2)

    cv2.show("Image", img)
    cv2.waitKey(0)

