import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_lines = mp.solutions.face_mesh_connections
pTime = 0

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

    results = face_mesh.process(image=imgRGB)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(image=img,
                               connections=mp_face_mesh.FACEMESH_TESSELATION,
                               landmark_drawing_spec=None,
                               landmark_list=face_landmarks,
                               connection_drawing_spec=mp_draw_styles.get_default_face_mesh_tesselation_style())
            mp_draw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_styles.get_default_face_mesh_contours_style())

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
