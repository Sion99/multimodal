import sys
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque, Counter
import threading
from screeninfo import get_monitors
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# PyAutoGUI fail-safe 비활성화
pyautogui.FAILSAFE = False

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 제스처 버퍼 설정
BUFFER_SIZE = 5
gesture_buffer = deque(maxlen=BUFFER_SIZE)

scroll_mode = False
zoom_mode = False
initial_distance = None
last_click_time = time.time()
last_gesture = None
dragging = False

# 최소 클릭 간격 설정
CLICK_INTERVAL = 0.3  # 클릭 간 최소 간격 (초)

prev_finger_pos = None  # 이전 손가락 위치

# 마우스 감도 설정
MOUSE_SENSITIVITY = 0.6  # 감도가 높을 수록 더 많이 이동
SCROLL_SENSITIVITY = 0.2  # 스크롤 감도

thumb_index_distance = 1

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
    global scroll_mode, thumb_index_distance
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

    # 스크롤 모드 판단
    thumb_tip = (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y)
    index_tip = (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y)
    thumb_index_distance = calculate_distance(thumb_tip, index_tip)

    if thumb_index_distance < 0.045:
        scroll_mode = True
    else:
        scroll_mode = False

    return fingers


def recognize_gesture(fingers_status):
    """
    손가락 상태를 기반으로 제스처를 인식하는 함수
    """
    if fingers_status == [0, 1, 0, 0, 0]:
        return 'move'
    elif fingers_status == [1, 1, 0, 0, 0]:
        return 'standby'
    elif scroll_mode:
        return 'scroll'
    elif fingers_status == [1, 0, 0, 0, 0]:
        return 'click'
    elif fingers_status == [1, 1, 1, 0, 0]:
        return 'drag'
    elif fingers_status == [1, 1, 1, 1, 1]:
        return 'move'

    return 'unknown'


def perform_mouse_action(gesture, lmList, handedness):
    global zoom_mode, initial_distance, dragging, prev_finger_pos, last_gesture, scroll_mode

    x, y = lmList[8].x, lmList[8].y  # 현재 손가락 위치
    mouse_x, mouse_y = pyautogui.position()

    wrist = (lmList[0].x, lmList[0].y)  # 손목 위치

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


class VideoCaptureWidget(QWidget):
    def __init__(self, parent=None):
        super(VideoCaptureWidget, self).__init__(parent)

        self.video_size = (640, 480)

        # OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)

        # PyQt5 Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update frame every 30ms

        # Set up the layout
        self.image_label = QLabel(self)
        self.image_label.resize(*self.video_size)

        # Gesture label for displaying gesture information
        self.gesture_label = QLabel(self)
        self.gesture_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.gesture_label)
        self.setLayout(layout)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                    mp_drawing.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            image = QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.gesture_label.setText(f"Gesture: {last_gesture}")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Hand Gesture Recognition")
        self.setGeometry(100, 100, 800, 600)
        self.video_widget = VideoCaptureWidget(self)
        self.setCentralWidget(self.video_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
