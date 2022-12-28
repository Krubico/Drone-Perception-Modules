import cv2
import os
import numpy as np
import dlib
import argparse
import face_recognition
import OpenCV_FaceDetection_Module

dir = 'Jia Hui'
xml_file = 'haar_face.xml'
people = ['Jia Hui', 'Unknown']
features = []
labels = []
train_encodings = []
known_people = []

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", type=str, required=True,
	#help="path to input image")
#ap.add_argument("-m", "--model", type=str,
	#default="mmod_human_face_detector.dat",
	#help="path to dlib's CNN face detector model")
#ap.add_argument("-u", "--upsample", type=int, default=1,
	#help="# of times to upsample")
#args = vars(ap.parse_args())

for file in os.listdir(dir):
    img_RGB = cv2.imread(f'{file}', 0)
    # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = FaceDetector()
    _, face_bbox = face_detector.findFaces(img=img_RGB, draw=False)
    x, y, w, h = face_bbox
    roi = img_gray[y: y+h, x:x+w]
    if f'{file_name}'.split()[0] in known_people:
        known_face_encodings.append(face_recognition.face_encodings(roi)[0])
    else:
        unknown_face_encodings.append(face_recognition.face_encodings(roi)[0])


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


