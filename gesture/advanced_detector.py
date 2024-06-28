import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyautogui
import math

detector = HandDetector(detectionCon=0.8)


def detect_hand_and_get_fingers(image):
    hands, img = detector.findHands(image, flipType=False)
    if hands:
        # 첫 번째 손 정보
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        fingers1 = detector.fingersUp(hand1)
        x1, y1 = lmList1[8][0], lmList1[8][1]
        x2, y2 = lmList1[12][0], lmList1[12][1]
        length1 = math.hypot(x2 - x1, y2 - y1)
        return fingers1, lmList1, length1, img
    return None, None, None, img


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


def calculate_zoom_distance(lmList):
    # 엄지와 검지 사이의 거리 계산
    x1, y1 = lmList[4][0], lmList[4][1]
    x2, y2 = lmList[8][0], lmList[8][1]
    return math.hypot(x2 - x1, y2 - y1)
