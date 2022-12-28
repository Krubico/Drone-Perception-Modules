import cv2
from retinaface.pre_trained_models import get_model

model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()

cap = cv2.VideoCapture(0)
success, img = cap.read()

while True:
    anno = model.predict_jsons(img)
    bbox = anno["bbox"]
    bbox = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
    cv2.rectangle(img, bbox, (255, 0, 255), rt)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
