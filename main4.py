import cv2
import mediapipe as mp
import numpy as np
from collections import Counter, deque
import threading
import pyautogui
import time

# 전역 변수 초기화
zoom_mode = False
initial_distance = None
pointer_stopped = False
standby_mode = False
last_action = None
prev_fingers1 = None
last_click_time = 0
CLICK_INTERVAL = 0.5  # 클릭 간 최소 간격 (초)
FRAME_THRESHOLD = 5  # 상태 전환을 위한 프레임 임계값
BUFFER_SIZE = 25  # 버퍼 크기

# 프레임 카운터 초기화
thumb_open_frames = 0
thumb_closed_frames = 0

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 제스처 버퍼 설정
gesture_buffer = deque(maxlen=BUFFER_SIZE)

def handle_mouse_events(action, lmList, cap):
    global pointer_stopped, zoom_mode, initial_distance, standby_mode

    if action == 'move' and not standby_mode:
        x, y = lmList[8][0], lmList[8][1]
        move(x, y, cap.get(3), cap.get(4))
    elif action == 'left click':
        pyautogui.click(button='left')
    elif action == 'right click':
        pyautogui.click(button='right')
    elif action == 'scroll':
        y_velocity = lmList[8][1] - lmList[7][1]
        if y_velocity > 0:
            pyautogui.scroll(-80)
        else:
            pyautogui.scroll(80)

def main():
    global zoom_mode, initial_distance, pointer_stopped, last_action, prev_fingers1, last_click_time, standby_mode
    global thumb_open_frames, thumb_closed_frames, gesture_buffer

    # 웹캠 설정
    cap = setup_webcam()
    previous_time = time.time()

    print("Webcam is running... Press 'ESC' to exit.")
    while True:
        try:
            img = read_frame(cap)  # 프레임 읽기
        except Exception as e:
            print(e)
            break  # 웹캠 읽기 실패시 루프 종료

        # 프레임 계산
        fps, delay, previous_time = calculate_fps_and_delay(previous_time)

        # 손동작 인식
        hand1_data = detect_hand_and_get_fingers(img)

        if hand1_data:
            fingers1, lmList1, length1, img = hand1_data
            if fingers1 is not None and lmList1 is not None:
                action1 = handle_gesture(fingers1, length1, prev_fingers1)

                if fingers1[0] == 1 and fingers1[1] == 1 and not standby_mode:
                    thumb_open_frames += 1
                    thumb_closed_frames = 0
                elif fingers1[0] == 0 and fingers1[1] == 1 and standby_mode:
                    thumb_closed_frames += 1
                    thumb_open_frames = 0
                else:
                    thumb_open_frames = 0
                    thumb_closed_frames = 0

                if thumb_open_frames >= FRAME_THRESHOLD:
                    standby_mode = True
                    thumb_open_frames = 0
                elif thumb_closed_frames >= FRAME_THRESHOLD:
                    standby_mode = False
                    thumb_closed_frames = 0

                # 제스처 버퍼에 추가
                gesture_buffer.append(action1)

                # 버퍼의 최빈값으로 제스처 결정
                if gesture_buffer:
                    most_common_gesture = Counter(gesture_buffer).most_common(1)[0][0]
                    perform_action(most_common_gesture, lmList1, cap)

                last_action = action1
                prev_fingers1 = fingers1

        cv2.putText(img, f"FPS: {int(fps)} DELAY: {delay}ms Action: {last_action} Standby: {standby_mode}",
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # img에 저장된 이미지를 Camera Feed라는 창에 출력함.
        cv2.imshow("Camera Feed", img)

        # 사용자가 "ESC"키를 누르면 루프를 종료함.
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_gesture(landmarks):
    # 각 손가락의 상태 (펼침: 1, 접음: 0)
    fingers = []

    # 엄지
    if landmarks[4][0] > landmarks[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # 나머지 네 손가락
    for i in [8, 12, 16, 20]:
        if landmarks[i][1] < landmarks[i - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    # 제스처 인식 (예: 주먹, 숫자, 특정 제스처 등)
    if fingers == [0, 0, 0, 0, 0]:
        return 'fist'
    elif fingers == [0, 1, 0, 0, 0]:
        return 'point'
    elif fingers == [1, 1, 1, 1, 1]:
        return 'open'
    elif fingers == [0, 1, 1, 0, 0]:
        return 'peace'
    # 필요한 다른 제스처를 추가

    return 'unknown'

def perform_action(gesture, lmList, cap):
    if gesture == 'fist':
        print('Detected Fist')
        # 주먹 제스처에 대한 동작 수행
    elif gesture == 'point':
        print('Detected Pointing')
        handle_mouse_events('move', lmList, cap)
    elif gesture == 'open':
        print('Detected Open Hand')
        # 열린 손 제스처에 대한 동작 수행
    elif gesture == 'peace':
        print('Detected Peace Sign')
        # 평화 제스처에 대한 동작 수행
    else:
        print('Unknown Gesture')

# Mediapipe 손 랜드마크 인식 및 관련 함수들
def detect_hand_and_get_fingers(image):
    global hands  # 전역 변수 hands 참조
    result = hands.process(image)
    if result.multi_hand_landmarks:
        # 첫 번째 손 정보
        hand1 = result.multi_hand_landmarks[0]
        lmList1 = [[lm.x, lm.y, lm.z] for lm in hand1.landmark]
        fingers1 = [int(lm.x > hand1.landmark[0].x) for lm in hand1.landmark[4:21:4]]
        x1, y1 = lmList1[8][0], lmList1[8][1]
        x2, y2 = lmList1[12][0], lmList1[12][1]
        length1 = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
        return fingers1, lmList1, length1, image
    return None, None, None, image

def setup_webcam(cam_id=0, width=320, height=180):
    cap = cv2.VideoCapture(cam_id)
    cap.set(3, width)
    cap.set(4, height)
    if not cap.isOpened():
        raise Exception("Failed to read the frame. Check the webcam connection.")
    return cap

def read_frame(cap):
    success, frame = cap.read()
    if not success:
        raise Exception("Failed to read the frame. Check the webcam connection.")
    frame = cv2.flip(frame, 1)  # 좌우 반전
    return frame

def calculate_fps_and_delay(previous_time):
    current_time = time.time()
    sec = current_time - previous_time
    fps = 1 / sec
    return int(fps), int(1000 / fps), current_time

def move(x, y, width, height):
    screen_width, screen_height = pyautogui.size()
    conv_x = int(np.interp(x, (0, width), (0, screen_width)))
    conv_y = int(np.interp(y, (0, height), (0, screen_height)))
    pyautogui.moveTo(conv_x, conv_y)

def handle_gesture(fingers, length, prev_fingers):
    action = None
    if fingers == [1, 1, 0, 0, 0]:
        action = "move"
    elif fingers == [0, 1, 0, 0, 0] and prev_fingers != fingers:
        action = "left click"
    elif fingers == [0, 1, 1, 0, 0] and prev_fingers != fingers:
        action = "right click"
    elif fingers == [1, 1, 1, 1, 1]:
        action = "scroll"
    return action

if __name__ == "__main__":
    main()
