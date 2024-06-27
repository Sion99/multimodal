# gesture/webcam.py
import cv2


def setup_webcam(cam_id=0, width=1280, height=720):
    cap = cv2.VideoCapture(cam_id)
    cap.set(3, width)
    cap.set(4, height)
    if not cap.isOpened():
        raise Exception("웹캠에서 이미지를 읽어오는 데 실패했습니다. 웹캠 연결상태를 체크하세요.")
    return cap


def read_frame(cap):
    success, frame = cap.read()
    if not success:
        raise Exception("웹캠에서 이미지를 읽어오는 데 실패했습니다. 웹캠 연결상태를 체크하세요.")
    frame = cv2.flip(frame, 1)  # 좌우 반전
    return frame


def calculate_fps(cap):
    return cap.get(cv2.CAP_PROP_FPS)
