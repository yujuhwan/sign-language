# 움직이는 제스처 인식 프로그램 데이터 저장(RNN-> LSTM)

import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['사랑해', '안녕하세요', '만나서', '반갑다'] # 학습 시킬 이미지 [0,1,2,3]에 매치
seq_length = 30  # 윈도우 사이즈
secs_for_action = 30  # 액션 녹화 시간: 30초 -> 안된다면 시간을 늘리던가 줄이자

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())  # time 함수 이용 시간 측정 (dataset 파일명 생성때 사용)
os.makedirs('dataset', exist_ok=True)  # os 함수이용 디렉토리 (dataset 폴더) 생성 / 데이터 셋 저장할 폴더 생성

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)  # 거울 반전

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        # 어떤 이미지를 녹화 할지
        cv2.imshow('img', img)
        cv2.waitKey(3000)  # 3초 대기

        start_time = time.time()  # dataset 생성 시간을 재기 위한 time 함수

        while time.time() - start_time < secs_for_action:  # 30초 동안 반복
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)  # 결과를 mediapipe에 넣어주기
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]  # visibility: 손가락 노드가 이미지 상에서 보이는지 판단

                    # 관절 사이의 각도 계산
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]  # 팔목과 각 손가락 관절사이의 벡터값
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)  # label 넣어주기 [ 0, 1, 2]

                    d = np.concatenate([joint.flatten(), angle_label])  # 100개 행렬

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)  # landmarks 그리기

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)  # 데이터를 numpy array 형태로 변환
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)  # raw 데이터 셋 파일 만들기

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)  # seq 데이터 셋 파일 만들기
    break




