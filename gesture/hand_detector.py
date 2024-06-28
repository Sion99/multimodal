# gesture/hand_detector.py
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyautogui
import math

detector = HandDetector(detectionCon=0.8)


def detect_hand_and_get_fingers(image):
    hands, img = detector.findHands(image, flipType=False)

    if len(hands) == 1:
        lmlist = hands[0]['lmList']
        x1, y1 = lmlist[8][0], lmlist[8][1]
        x2, y2 = lmlist[12][0], lmlist[12][1]
        length = math.hypot(x2 - x1, y2 - y1)
        fingers = detector.fingersUp(hands[0])
        return fingers, (x1, y1), length, img
    return None, (None, None), None, img


def move(x, y, width, height):
    screen_width, screen_height = pyautogui.size()
    conv_x = int(np.interp(x, (0, width), (0, screen_width)))
    conv_y = int(np.interp(y, (0, height), (0, screen_height)))
    pyautogui.moveTo(conv_x, conv_y)


def handle_gesture(previous_fingers, fingers):
    if previous_fingers == [0, 1, 0, 0, 0] and fingers == [0, 0, 0, 0, 0]:
        pyautogui.click(button='left')
        return "left click"
    if previous_fingers == [0, 1, 1, 0, 0] and fingers == [0, 0, 0, 0, 0]:
        pyautogui.click(button='right')
        return "right click"
    


def handle_gesture(fingers, length):
    if fingers == [1, 1, 0, 0, 0]:
        return "move"
    elif fingers == [0, 1, 1, 1, 1]:
        pyautogui.scroll(80)
        return "scroll up"
    elif fingers == [1, 0, 0, 0, 0]:
        pyautogui.scroll(-80)
        return "scroll down"
    elif fingers == [0, 1, 0, 0, 1]:
        pyautogui.click(button='left')
        return "left click"
    elif fingers == [1, 1, 0, 0, 1]:
        pyautogui.click(button='right')
        return "right click"
    elif fingers == [0, 1, 1, 1, 1]:
        pyautogui.doubleClick(button='left')
        return "double click"
    elif fingers == [1, 1, 1, 0, 0] and length < 30:
        pyautogui.mouseDown(button='left')
        return "drag"
    return None

# def handle_gesture(fingers, length, voice_label):
#     if fingers == [1, 1, 0, 0, 0]:
#         return "move"
#     elif fingers == [0, 1, 1, 1, 1] and voice_label == 'up':
#         pyautogui.scroll(80)
#         return "scroll up"
#     elif fingers == [1, 0, 0, 0, 0] and voice_label == 'down':
#         pyautogui.scroll(-80)
#         return "scroll down"
#     elif fingers == [0, 1, 0, 0, 1] and voice_label == 'left':
#         pyautogui.click(button='left')
#         return "left click"
#     elif fingers == [1, 1, 0, 0, 1] and voice_label == 'right':
#         pyautogui.click(button='right')
#         return "right click"
#     elif fingers == [0, 1, 1, 1, 1] and voice_label == 'two':
#         pyautogui.doubleClick(button='left')
#         return "double click"
#     elif fingers == [1, 1, 1, 0, 0] and length < 30:
#         pyautogui.mouseDown(button='left')
#         return "drag"
#     return None
