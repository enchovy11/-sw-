import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

cap = cv2.VideoCapture("/home/master/curv.mp4")


def preprocess_traffic_light(image):
    image = cv2.resize(image, (0, 0), fx=2, fy=2)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return image


def classify_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)

    if red_pixels > green_pixels and red_pixels > 50:
        return "Red"
    elif green_pixels > red_pixels and green_pixels > 50:
        return "Green"
    else:
        return "Unknown"


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for *xyxy, conf, cls in results.xyxy[0]:
        if model.names[int(cls)] == 'traffic light':
            x1, y1, x2, y2 = map(int, xyxy)
            traffic_light_img = frame[y1:y2, x1:x2]

            processed_img = preprocess_traffic_light(traffic_light_img)
            color_class = classify_color(processed_img)

            label = f"Traffic Light: {color_class} {conf:.2f}"
            color = (0, 255, 0) if color_class == 'Green' else (0, 0, 255) if color_class == 'Red' else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Traffic Light Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
