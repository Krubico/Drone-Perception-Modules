import os
import cv2
import mediapipe as mp
import face_recognition
import dlib
import numpy as np
import pickle
from OpenCV_FaceDetection_Module import FaceDetector
from PIL import Image
import cProfile, pstats, io
import multiprocessing as mp


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class FaceRecognition():
    def __init__(self):
        self.face_recognition = face_recognition
        self.face_encodings = self.face_recognition.face_encodings
    @profile
    def get_faceEncoding(self, images, face_location=[]):
        encodings = []
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        encoded_img = self.face_encodings(images, known_face_locations=[face_location])[0]
        encodings.append(encoded_img)
        return encodings
    def faceRecognizer(self, img, encodings, classNames=[]):
        name = ''
        face_recogs = self.face_encodings(img)
        #face_locations = face_recognition.face_locations(image, model="cnn")

        for face_encode in zip(face_recogs):
            matches = self.face_recognition.compare_faces(encodings, face_encode)
            faceDistance = self.face_recognition.face_distance(encodings, face_encode)
            faceDistance = faceDistance.tolist()
            matchIndex = faceDistance.index(min(faceDistance))
            print(matchIndex)

            if matches[matchIndex][0]:
                name = classNames[matchIndex]

                #for i in matchedIdxs:
                    #name = data["names"][i]
                    #counts[name] = counts.get(name, 0) + 1
        return name

@profile
def imgshow(img):
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)


@profile
def draw(img, face_bboxs, name):
    for i in range(len(face_bboxs)):
        for (x, y, w, h) in face_bboxs[i][1]:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 1)
            cv2.putText(img, f'{name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


def test():
    face_detector = FaceDetector()
    face_recognizer = FaceRecognition()
    known_names = ['Jia Hui']
    unknown_names = ['unknown']
    classNames = []
    encodings = []

    for file in os.listdir('Face_Recognition_images'):
        file_img = Image.open(f'Face_Recognition_images/{file}')
        file_img = np.asarray(file_img)
        # cv2.imshow("Photo", file)
        _, face_bboxs = face_detector.findFaces(file_img, draw=False)
        file_imggray = cv2.cvtColor(file_img, cv2.COLOR_BGR2GRAY)

        #for (x, y, w, h) in face_bboxs[0][1]:
        if face_bboxs:
            bbox = face_bboxs[0][1]
            x, y, w, h = bbox[0]
            # roi = file_imggray[y: y + h, x:x + w]
            roi = file_img[y: y + h, x:x + w]
            encoding = face_recognizer.get_faceEncoding(roi, face_location=[0, w, h, 0])
            encodings.append(encoding)
        # print(encodings)

        if f'{file}' in known_names:
            classNames.append(f'{file}'.split()[0])
        else:
            classNames.append('Unknown')
    encodings = np.array(encodings)

    cap = cv2.VideoCapture(0)
    #Singular Face Recognizer
    while True:
        success, img = cap.read()
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, face_bboxs = face_detector.findFaces(img_RGB, draw=False)
        roi = img_RGB[y: y+h, x:x+w]

        name = face_recognizer.faceRecognizer(roi, encodings=encodings, classNames=classNames)

        if face_bboxs:
            for i in range(len(face_bboxs)):
                for (x, y, w, h) in face_bboxs[i][1]:
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 1)
                    cv2.putText(img, f'{name}', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Webcam", img)
        # cv2.waitKey(0)





test()

def train():
    encodings = faceEncoding(images)

    path = 'Train_Images'
    images = []
    classNames = []
    listDir = os.listdir(path)
    for cls in listDir:
        img = cv2.imread(f'{path}/{cls}')
        images.append(img)
        classNames.append(cls.split()[0])

    detector = FaceRecognition()
    recog_face = detector.faceRecogizer(img)
