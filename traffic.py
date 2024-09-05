import torch
import cv2
import numpy as np
import time

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

# 비디오 캡처
cap = cv2.VideoCapture("/home/master/curv.mp4")

# 원본 영상의 FPS 가져오기
original_fps = cap.get(cv2.CAP_PROP_FPS)
# 재생 속도를 조절하기 위한 factor (예: 2배 느리게)
slowdown_factor = 3

frame_skip = 3  # 3개의 프레임마다 1개씩 처리
frame_count = 0

# 프레임 간 시간 간격 계산 (밀리초 단위)
frame_interval = int((1000 / original_fps) * slowdown_factor)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # 프레임 건너뛰기

    # YOLOv5로 객체 감지
    results = model(frame)

    # 결과를 데이터프레임으로 변환
    df = results.pandas().xyxy[0]

    # 신호등만 필터링
    traffic_lights = df[df['name'] == 'traffic light']

    # 감지된 신호등에 바운딩 박스 그리기
    for _, light in traffic_lights.iterrows():
        x1, y1, x2, y2 = map(int, [light['xmin'], light['ymin'], light['xmax'], light['ymax']])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Traffic Light {light['confidence']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 표시
    cv2.imshow('Traffic Light Detection', frame)

    # 계산된 프레임 간격만큼 대기
    if cv2.waitKey(frame_interval) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
