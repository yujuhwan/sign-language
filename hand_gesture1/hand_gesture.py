# CVZone 패키지를 활용한 제스처 인식으로 동영상 플레이어

from cvzone.HandTrackingModule import HandDetector  # cvzone 패키지에서 HandDetector만 사용
import cv2

detector = HandDetector(maxHands=1)  # 손 개수 인식

cap_cam = cv2.VideoCapture(0)  # 웹캠
cap_video = cv2.VideoCapture('video.mp4')  # 동영상 출력

w = int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # 카메라 프레임의 가로 길이

total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT)) # 비디오의 전체 프레임 개수: 241개
print(total_frames)

_, video_img = cap_video.read()  # 첫번째 프레임을 읽음. 정지 상태

# 타임라인 그리기
def draw_timeline(video_img, rel_x):
    img_h, img_w, img_c = video_img.shape
    timeline_w = int(img_w * rel_x)  # 타임라인의 너비
    cv2.rectangle(video_img, pt1=(0, img_h - 50), pt2=(timeline_w, img_h - 48), color=(0, 0, 255), thickness=-1)

# 타임라인과 변수 초기화
rel_x = 0
frame_idx = 0
draw_timeline(video_img, rel_x)

while cap_cam.isOpened():   # 카메라가 열리면
    ret, cam_img = cap_cam.read()  # 카메라 프레임을 한 프레임씩 읽기

    if not ret:
        break

    cam_img = cv2.flip(cam_img, 1)  # 거울 반전

    hands, cam_img = detector.findHands(cam_img)  # 손의 랜드마크 찾기

    if hands:  # 카메라에 손이 인식되면
        lm_list = hands[0]['lmList']  # 손의 랜드마크 리스트를 받는데
        fingers = detector.fingersUp(hands[0])  # 손가락을 접으면 0, 피면 1 (5개로 반환)

        # cv2.putText(cam_img, text=str(fingers), org=(10, 120),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=2)

        length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img) # 엄지, 검지 사이의 거리를 계산

        if fingers == [0, 0, 0, 0, 0]:  # 주먹을 쥐고 있을 때 정지 모드
            pass
        else:  # 탐색 또는 플레이 모드
            if length < 30:  # Navigate 탐색
                rel_x = lm_list[4][0] / w  # 엄지손가락의 x좌표를 상대좌표 0~1 좌표로 변환

                frame_idx = int(rel_x * total_frames)  #  엄지손가락 x 좌표에 따른 동영상 플레임 번호 계산
                if frame_idx < 0:
                    frame_idx = 0
                elif frame_idx > total_frames:
                    frame_idx = total_frames

                cap_video.set(1, frame_idx)  # 동영상을 해당 프레임(frame_idx)로 이동

                cv2.putText(cam_img, text='Navigate %.2f, %d' % (rel_x, frame_idx), org=(10,50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2, color=(255,255,255), thickness=2)

            else:  # play 재생
                frame_idx = frame_idx + 1  # 다음 프레임 재생
                rel_x = frame_idx / total_frames

            if frame_idx < total_frames:  # 예외 처리
                _, video_img = cap_video.read()  # 동영상의 프레임을 읽음
                draw_timeline(video_img, rel_x)  # 타임라인을 그린다

    cv2.imshow('cam', cam_img)
    cv2.imshow('video', video_img)

    if cv2.waitKey(1) == ord('q'):
        break

