import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque, Counter
import threading
import webcam
from screeninfo import get_monitors

# PyAutoGUI fail-safe 비활성화
pyautogui.FAILSAFE = False

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 제스처 버퍼 설정
BUFFER_SIZE = 25
gesture_buffer = deque(maxlen=BUFFER_SIZE)

zoom_mode = False
initial_distance = None
last_click_time = time.time()
last_gesture = None
dragging = False

# 최소 클릭 간격 설정
CLICK_INTERVAL = 0.3  # 클릭 간 최소 간격 (초)

prev_finger_pos = None  # 이전 손가락 위치

# 마우스 감도 설정
MOUSE_SENSITIVITY = 0.7  # 감도가 높을 수록 더 많이 이동
SCROLL_SENSITIVITY = 0.3  # 스크롤 감도

# 전체 모니터 해상도 계산
monitors = get_monitors()
total_screen_width = sum(monitor.width for monitor in monitors)
total_screen_height = max(monitor.height for monitor in monitors)


def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def get_finger_status(hand_landmarks):
    """
    손가락이 펴져 있는지 접혀 있는지 확인하는 함수
    """
    fingers = []

    # 엄지
    if hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x:  # 왼손
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # 오른손
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # 나머지 손가락
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    for tip, pip in zip(tips, pip_joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def recognize_gesture(fingers_status):
    """
    손가락 상태를 기반으로 제스처를 인식하는 함수
    """
    if fingers_status == [0, 1, 0, 0, 0]:
        return 'move'
    elif fingers_status == [1, 1, 0, 0, 0]:
        return 'standby'
    elif fingers_status == [1, 0, 0, 0, 0]:
        return 'click'
    elif fingers_status == [1, 1, 1, 0, 0]:
        return 'drag'
    elif fingers_status == [0, 1, 1, 1, 1]:
        return 'scroll'
    elif fingers_status == [1, 1, 1, 1, 1]:
        return 'move'
    return 'unknown'


def perform_mouse_action(gesture, lmList, handedness):
    global zoom_mode, initial_distance, dragging, prev_finger_pos, last_gesture

    x, y = lmList[8].x, lmList[8].y  # 현재 손가락 위치
    mouse_x, mouse_y = pyautogui.position()

    if prev_finger_pos is not None:
        # 상대 이동 거리 계산
        dx = (x - prev_finger_pos[0]) * total_screen_width * MOUSE_SENSITIVITY
        dy = (y - prev_finger_pos[1]) * total_screen_height * MOUSE_SENSITIVITY

        # 마우스 포인터가 화면 가장자리에 도달할 때 상대 이동 보정
        if mouse_x + dx < 0:
            dx = -mouse_x
        elif mouse_x + dx > total_screen_width:
            dx = total_screen_width - mouse_x
        if mouse_y + dy < 0:
            dy = -mouse_y
        elif mouse_y + dy > total_screen_height:
            dy = total_screen_height - mouse_y

        if gesture == 'move':
            pyautogui.moveRel(dx, dy)
        elif gesture == 'drag':
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
            else:
                pyautogui.moveRel(dx, dy)
        elif gesture == 'scroll':
            if handedness == 'Left':
                if lmList[4].x < lmList[20].x:
                    pyautogui.scroll(dy * SCROLL_SENSITIVITY)
                else:
                    pyautogui.scroll(-dy * SCROLL_SENSITIVITY)
            else:
                if lmList[4].x > lmList[20].x:
                    pyautogui.scroll(dy * SCROLL_SENSITIVITY)
                else:
                    pyautogui.scroll(-dy * SCROLL_SENSITIVITY)
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False

    prev_finger_pos = (x, y)
    last_gesture = gesture


def perform_click_action(gesture):
    global last_click_time, last_gesture

    current_time = time.time()
    if gesture == 'click' and gesture != last_gesture and current_time - last_click_time > CLICK_INTERVAL:
        pyautogui.click()
        last_click_time = current_time

    last_gesture = gesture


def main():
    global zoom_mode, initial_distance, last_click_time, prev_finger_pos, MOUSE_SENSITIVITY, SCROLL_SENSITIVITY

    cap = webcam.setup_webcam()
    previous_time = time.time()

    print("Webcam is running... Press 'ESC' to exit.")
    while True:
        try:
            img = webcam.read_frame(cap)  # 프레임 읽기
        except Exception as e:
            print(e)
            break

        fps, delay, previous_time = webcam.calculate_fps_and_delay(previous_time)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                fingers_status = get_finger_status(hand_landmarks)
                gesture = recognize_gesture(fingers_status)
                gesture_buffer.append(gesture)

                # 버퍼의 최빈값으로 제스처 결정
                most_common_gesture = Counter(gesture_buffer).most_common(1)[0][0]

                handedness = result.multi_handedness[i].classification[0].label

                if most_common_gesture in ['move', 'drag', 'scroll']:
                    thread = threading.Thread(target=perform_mouse_action,
                                              args=(most_common_gesture, hand_landmarks.landmark, handedness))
                    thread.start()
                else:
                    perform_click_action(most_common_gesture)

                # 손 랜드마크와 연결선 그리기
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(img,
                    f"F: {int(fps)} D: {delay}ms ACT: {last_gesture} MSENS: {round(MOUSE_SENSITIVITY, 3)} SCR: {round(SCROLL_SENSITIVITY, 3)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture', img)

        input = cv2.waitKey(1)

        if input == 27:
            break
        elif input == ord('a'):
            if MOUSE_SENSITIVITY > 0.1:
                MOUSE_SENSITIVITY -= 0.1
        elif input == ord('s'):
            if MOUSE_SENSITIVITY < 2.9:
                MOUSE_SENSITIVITY += 0.1
        elif input == ord('d'):
            if SCROLL_SENSITIVITY > 0.1:
                SCROLL_SENSITIVITY -= 0.05
        elif input == ord('f'):
            if SCROLL_SENSITIVITY < 0.9:
                SCROLL_SENSITIVITY += 0.05

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
