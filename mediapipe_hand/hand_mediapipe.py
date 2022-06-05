# MediaPipe를 활용한 손 인식 프로그램

import numpy as np
import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)

# MediaPipe Hands 손의 관절 위치를 인식할 수 있는 모델을 초기화한다
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 모델 초기화
hands = mp_hands.Hands(
    max_num_hands=3,  # 쵀대 손의 인식 개수
    min_detection_confidence=0.5,  # 타미 임계치
    min_tracking_confidence=0.5,  # 추적 임계치
)

while cap.isOpened():  # 카메라가 열려있으면
    ret, img = cap.read()  # 카메라의 프레임을 한 프레임씩 읽는다
    if not ret:
        break

    img = cv2.flip(img, 1)  # 거울 반전
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    result = hands.process(img)  # 웹캠 이미지(프레임)에서 손의 위치, 관절 위치를 탐지한다.

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR

    if result.multi_hand_landmarks is not None:  # 손 인식 (손 하나면 한개의 리스트 저장, 여러개면 여러개 리스트 저장)
        for res in result.multi_hand_landmarks:  # 손이 여러개일수도 있으니 루프를 사용
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)  # 손을 이미지에 그림

    cv2.imshow('result', img)
    if cv2.waitKey(1)  == ord('q'):
        break