#!/usr/bin/env python

import cv2
from gesture import webcam, hand_detector
from voice import voice_recognition, voice_command


def main():
    # 웹캠 설정
    cap = webcam.setup_webcam()

    # 음성 인식 프로세스 시작
    voice_process = voice_command.start_voice_recognition()

    print("Webcam is running... Press 'ESC' to exit.")
    while True:
        try:
            img = webcam.read_frame(cap)  # 프레임 읽기
        except Exception as e:
            print(e)
            break  # 웹캠 읽기 실패시 루프 종료

        # 음성 명령 가져오기
        voice_label = voice_command.get_voice_cmd()

        # 손동작 인식
        fingers, (x, y), length, img = hand_detector.detect_hand_and_get_fingers(img)

        if fingers:
            action = hand_detector.handle_gesture(fingers, length, voice_label)
            if action == 'move':
                hand_detector.move_mouse(x, y, cap.get(3), cap.get(4))
            elif action == 'click':
                voice_command.reset_voice_label()
            print(f"Action: {action}")

        # img에 저장된 이미지를 Camera Feed라는 창에 출력함.
        cv2.imshow("Camera Feed", img)

        # 사용자가 "ESC"키를 누르면 루프를 종료함.
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    voice_process.terminate()


if __name__ == "__main__":
    main()
