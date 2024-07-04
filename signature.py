#!/usr/bin/env python

import cv2
from gesture import webcam, advanced_detector
import numpy as np
import time
import threading

drawing = False
last_position = None
signature = None
signature_lock = threading.Lock()


def handle_drawing(lmList1):
    global drawing, last_position, signature

    with signature_lock:
        if drawing and lmList1 is not None:
            current_position = (lmList1[8][0], lmList1[8][1])
            if last_position:
                cv2.line(signature, last_position, current_position, (0, 0, 0), 5)
            last_position = current_position


def main():
    global drawing, last_position, signature

    # 웹캠 설정
    cap = webcam.setup_webcam()
    previous_time = time.time()

    print("Webcam is running... Press 'ESC' to exit.")
    while True:
        try:
            _ = webcam.read_frame(cap)  # 프레임 읽기, 이미지는 사용하지 않음
        except Exception as e:
            print(e)
            break  # 웹캠 읽기 실패시 루프 종료

        # 손동작 인식
        hand1_data = advanced_detector.detect_hand_and_get_fingers(_)

        # 흰 배경 이미지 생성
        img = np.ones((480, 720, 3), dtype=np.uint8) * 255

        if hand1_data:
            fingers1, lmList1, length1, _ = hand1_data

            # 서명 스레드 처리
            drawing_thread = threading.Thread(target=handle_drawing, args=(lmList1,))
            drawing_thread.start()

            if lmList1 is not None:
                # 손가락 위치 표시
                cv2.circle(img, (lmList1[8][0], lmList1[8][1]), 10, (0, 0, 255), -1)

        # 서명 이미지와 현재 프레임 결합
        with signature_lock:
            if signature is not None:
                combined = cv2.addWeighted(img, 0.5, signature, 0.5, 0)
            else:
                combined = img

        # img에 저장된 이미지를 Display라는 창에 출력함.
        cv2.imshow("Sign here", combined)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 키를 누르면 루프를 종료함.
            break
        elif key == 32:  # 스페이스바를 누르면 서명 시작/중지
            drawing = not drawing
            if drawing:
                # 서명을 위한 3채널 이미지 생성
                signature = np.ones((480, 720, 3), dtype=np.uint8) * 255  # 흰 배경
            last_position = None
        elif key == 13:  # 엔터 키를 누르면 서명 완료 및 저장
            if signature is not None:
                timestamp = int(time.time())
                cv2.imwrite(f"signature_{timestamp}.png", signature)
                print(f"Signature saved as signature_{timestamp}.png")
                drawing = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()