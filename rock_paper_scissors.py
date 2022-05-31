# 가위바위보 인식 프로그램

import tensorflow.keras
import numpy as np
import cv2

model = tensorflow.keras.models.load_model('keras_model.h5')  # 티처블머신 모델 load

cap = cv2.VideoCapture(0)  # 동영상 키는법: ('동영상 경로')

classes = ['Scissors', ' Rock', 'Paper']

while cap.isOpened():  # 열려있는 동안 while문 실행
    ret, img = cap.read()  # 한 프레임씩 읽는 법, ret: 성공적으로 읽었는지(불리언), img: 각 프레임의 이미지를 numpy 형태로 반환

    if not ret:  # 읽는 데 실패 했을 경우
        break

    img = cv2.flip(img, 1)  # 이미지 좌우 반전:1

    h, w, c = img.shape  # h = 480, w = 640

    img = img[:, 160:160+h]  # 정사각형으로 자르기(티쳐블 머신 출력영상과 맞추기위해)

    img_input = cv2.resize(img, (224,224))  # 티처블 머신 해상도랑 동일값 이미지 (224,224) 사이즈로 변경
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변경  티처블 머신은 RGB를 사용
    # opencv에서는 0~255값이지만 티처블머신에서는 -1 ~ 1 값이므로 변경
    img_input = (img_input.astype(np.float32) / 127.0) - 1.0  # 전처리
    img_input = np.expand_dims(img_input, axis=0)  # 텐서플로우 모델에 넣으려면 차원의 개수 늘려야함. 0번 축을 추가 (1, 224, 224, 3)

    prediction = model.predict(img_input)  # 예측값(확률값)
    print(prediction)  # [[1.1941064e-01 8.8011581e-01 4.7359432e-04]] 가위 바위 보(확률)

    idx = np.argmax(prediction)  # 가장 높은 확률 idx에 저장

    # 출력하기
    cv2.putText(img, text=classes[idx], org=(10,50),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255), thickness=2)

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):  # q버튼 누르면 꺼짐
        break

