# MediaPipie를 활용한 가위바위보 프로그램

import numpy as np
import mediapipe as mp
import cv2

gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}

rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

cap = cv2.VideoCapture(0)

# MediaPipe Hands 손의 관절 위치를 인식할 수 있는 모델을 초기화한다
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 모델 초기화
hands = mp_hands.Hands(
    max_num_hands=1,  # 쵀대 손의 인식 개수
    min_detection_confidence=0.5,  # 타미 임계치
    min_tracking_confidence=0.5,  # 추적 임계치
)

file = np.genfromtxt('../sign_language/gesture_train.csv', delimiter=',')  # 파일 읽어오기
angle = file[:, :-1].astype(np.float32)  # 0번 인덱스부터 마지막 인덱스(-1) 전까지 잘라서 가져옴 -> feature
label = file[:, -1].astype(np.float32)  # 마지막 인덱스(-1)만 가져옴 -> label

knn = cv2.ml.KNearest_create()  # KNN 모델 초기화
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # KNN 학습

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
            joint = np.zeros((21,3))
            for j, lm in enumerate(res.landmark):  # 21개의 랜드마크 중 한 점씩 반복문을 사용해서 처리한다 (좌표는 상대좌표, 0~1)
                joint[j] = [lm.x, lm.y, lm.z]

            # 관절 사이의 각도 계산
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1 # (20, 3) # 팔목과 각 손가락 관절사이의 벡터값

            v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=1)
            # (20,3) / (20, 1) 유닛 벡터값, # (벡터 / 벡터의 길이)

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]
            # 유닛벡터를 내적한 값의 아크코사인을 구하면 관절 사이의 각도를 구할 수 있다.

            angle = np.degrees(angle)  # 라디안 -> 도, 64bit, 각도 값으로 환산, gesture_train.csv의 앞에 값과 동일
            angle = np.expand_dims(angle.astype(np.float32), axis=0)  # 머신러닝 모델은 32bit이므로 변환
            # 머신러닝 모델에 넣어서 추론할 떄에는 항상 맨 앞 차원 하나를 추가해야 함.

            # 제스처 추론
            _, results, _, _ =  knn.findNearest(angle, 3)  # k = 3

            idx = int(results[0][0])  # 인덱스 정수 형태로 변환

            if idx in rps_gesture.keys():  # 가위, 바위, 보 중에 하나면 출력
                gesture_name = rps_gesture[idx]

                cv2.putText(img, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,0), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)  # 손을 이미지에 그림

    cv2.imshow('result', img)
    if cv2.waitKey(1)  == ord('q'):
        break
