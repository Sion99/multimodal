#!/usr/bin/env python

import cv2
import pyautogui
from gesture import webcam, advanced_detector
import time

zoom_mode = False
initial_distance = None
pointer_stopped = False


def main():
    global zoom_mode, initial_distance, pointer_stopped

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
        cv2.putText(img, f"FPS: {int(fps)} DELAY: {delay}ms", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # 손동작 인식
        hand1, hand2 = advanced_detector.detect_hand_and_get_fingers(img)

        if hand1:
            fingers1, lmlist1, length1, img = hand1
            action1 = advanced_detector.handle_gesture(fingers1, length1)

            if not pointer_stopped:
                if action1 == 'move':
                    x1, y1 = lmlist1[8][0], lmlist1[8][1]
                    advanced_detector.move(x1, y1, cap.get(3), cap.get(4))
                print(f"Hand 1 Action: {action1}")

            # 확대/축소 모드 진입
            if not zoom_mode and fingers1 == [1, 1, 0, 0, 0]:
                zoom_mode = True
                initial_distance = advanced_detector.calculate_zoom_distance(lmlist1)
                print("Zoom mode activated")

            # 확대/축소 모드에서 거리 계산
            if zoom_mode:
                current_distance = advanced_detector.calculate_zoom_distance(lmlist1)
                if initial_distance is not None:
                    if current_distance > initial_distance:
                        pyautogui.hotkey('ctrl', '+')
                        print("Zooming in")
                    elif current_distance < initial_distance:
                        pyautogui.hotkey('ctrl', '-')
                        print("Zooming out")
                    initial_distance = current_distance

            # 확대/축소 모드 종료
            if fingers1 != [1, 1, 0, 0, 0]:
                zoom_mode = False
                initial_distance = None
                print("Zoom mode deactivated")

        if hand2:
            fingers2, lmlist2, length2, img = hand2
            print(f"Hand 2 detected with fingers: {fingers2}")

            # 두 번째 손이 등장하면 마우스 포인터 정지
            pointer_stopped = True

            # 정지 상태에서 특정 제스처 인식
            if fingers2 == [0, 1, 0, 0, 0]:
                pyautogui.click(button='left')
                print("Left click")
            elif fingers2 == [0, 1, 1, 0, 0]:
                pyautogui.click(button='right')
                print("Right click")

        else:
            pointer_stopped = False

        # img에 저장된 이미지를 Camera Feed라는 창에 출력함.
        cv2.imshow("Camera Feed", img)

        # 사용자가 "ESC"키를 누르면 루프를 종료함.
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
