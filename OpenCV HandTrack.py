import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

HandVariables = {
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.6
}

#Getting hand landmarks
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode= False,
                       min_detection_confidence= HandVariables["min_detection_confidence"],
                       min_tracking_confidence= HandVariables["min_tracking_confidence"])
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handMarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handMarks.landmark):
                h, w, channels = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (0, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handMarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)



if __name__== "__main__":
    main()
