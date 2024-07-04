# gesture/webcam.py
import cv2
import time


def setup_webcam(cam_id=0, width=1280, height=720):
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
    fps = 1/sec
    return int(fps), int(1000 / fps), current_time
