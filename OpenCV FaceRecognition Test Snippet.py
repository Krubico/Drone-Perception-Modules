import cv2
import numpy as np
import face_recognition
from OpenCV FaceDetection Module import FaceDetector

cap = cv2.VideoCapture(0)
success, img = cv2.read()


if success:
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = FaceDetector()
    _, face_bbox = face_detector.findFaces(img=img_RGB, draw=False)
    x, y, w, h = face_bbox
    roi = img_gray[y: y+h, x:x+w]
    results = face_recognition.compare_faces([known_face_encodings], test_encoding)
    print(f'Hello {results}')

    cv.putText(img, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 255), thickness=2)
    cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), thickness=2)

    cv.imshow('Detected Face', img)
    cv.waiKey(1)


