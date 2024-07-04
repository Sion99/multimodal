#!/usr/bin/env python

import cv2
from gesture import webcam, advanced_detector
import time
import threading
import pyautogui

zoom_mode = False
initial_distance = None
pointer_stopped = False
last_action = None
prev_fingers1 = None
last_click_time = 0
CLICK_INTERVAL = 0.5  # 클릭 간 최소 간격 (초)



def handle_mouse_events(action, lmList, cap):
    global pointer_stopped, zoom_mode, initial_distance

    if action == 'move' and not pointer_stopped:
        x, y = lmList[8][0], lmList[8][1]
        advanced_detector.move(x, y, cap.get(3), cap.get(4))
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
    global zoom_mode, initial_distance, pointer_stopped, last_action, prev_fingers1, last_click_time

    # 웹캠 설정
    cap = webcam.setup_webcam()
    previous_time = time.time()

    print("Webcam is running... Press 'ESC' to exit.")
    while True:
        try:
            img = webcam.read_frame(cap)  # 프레임 읽기
        except Exception as e:
            print(e)
            break  # 웹캠 읽기 실패시 루프 종료

        # 프레임 계산
        fps, delay, previous_time = webcam.calculate_fps_and_delay(previous_time)

        # 손동작 인식
        hand1_data = advanced_detector.detect_hand_and_get_fingers(img)

        if hand1_data:
            fingers1, lmList1, length1, img = hand1_data
            action1 = advanced_detector.handle_gesture(fingers1, length1, prev_fingers1)

            # 모든 동작을 스레드로 처리
            current_time = time.time()
            if action1 == 'move' or (current_time - last_click_time > CLICK_INTERVAL):
                last_click_time = current_time
                if action1:
                    mouse_thread = threading.Thread(target=handle_mouse_events, args=(action1, lmList1, cap))
                    mouse_thread.start()

            last_action = action1
            prev_fingers1 = fingers1

        cv2.putText(img, f"FPS: {int(fps)} DELAY: {delay}ms Action: {last_action}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0), 2)
        # img에 저장된 이미지를 Camera Feed라는 창에 출력함.
        cv2.imshow("Camera Feed", img)

        # 사용자가 "ESC"키를 누르면 루프를 종료함.
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
