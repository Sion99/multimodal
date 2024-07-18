import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QStackedWidget, QAction, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from gesture.webcam import setup_webcam, read_frame
import time


class LoadingWidget(QWidget):
    loading_complete = pyqtSignal()

    def __init__(self, parent=None):
        super(LoadingWidget, self).__init__(parent)
        self.initUI()
        self.cap = None

    def initUI(self):
        self.label = QLabel("Loading camera, please wait...", self)
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Start loading process
        QTimer.singleShot(1500, self.load_camera)

    def load_camera(self):
        self.cap = setup_webcam()
        if self.cap.isOpened():
            self.loading_complete.emit()
        else:
            self.label.setText("Failed to load camera.")


class VideoCaptureWidget(QWidget):
    frame_received = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(VideoCaptureWidget, self).__init__(parent)

        self.video_size = (320, 240)  # 비디오 크기를 더 작게 설정

        # OpenCV Video Capture
        self.cap = setup_webcam()
        self.previous_time = time.time()

        # PyQt5 Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # Update frame every 10ms

        # Set up the layout
        self.image_label = QLabel(self)
        self.image_label.resize(*self.video_size)

        # Status label for displaying FPS, delay, and other information
        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def update_frame(self):
        frame = read_frame(self.cap)
        if frame is not None:
            self.frame_received.emit(True)  # Frame successfully received
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio))

            # Update status information
            current_time = time.time()
            fps = 1 / (current_time - self.previous_time)
            delay = (current_time - self.previous_time) * 1000  # in milliseconds
            self.previous_time = current_time
            scroll_sensitivity = 0.2  # Example value, replace with your variable

            self.status_label.setText(
                f"FPS: {fps:.2f}\n"
                f"Delay: {delay:.2f} ms\n"
                f"Scroll Sensitivity: {scroll_sensitivity:.2f}"
            )
        else:
            self.frame_received.emit(False)  # Frame not received

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

    def resizeEvent(self, event):
        if self.cap.isOpened():
            self.update_frame()


class MainWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super(MainWidget, self).__init__(parent)
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setLayout(QHBoxLayout())
        self.gesture_mode_checkbox = QCheckBox("Gesture Mode", self)
        self.gesture_mode_checkbox.stateChanged.connect(self.toggle_gesture_mode)
        self.layout().addWidget(self.gesture_mode_checkbox)

    def toggle_gesture_mode(self, state):
        if state == Qt.Checked:
            self.main_window.statusBar().showMessage('Gesture Mode On')
        else:
            self.main_window.statusBar().showMessage('Gesture Mode Off')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Hiperwall Multimodal Interface')
        self.setGeometry(100, 100, 800, 600)  # 초기 창 크기를 조정
        # self.setMinimumSize(320, 240)  # 최소 크기 설정

        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        self.loading_widget = LoadingWidget(self)
        self.stacked_widget.addWidget(self.loading_widget)

        self.video_widget = VideoCaptureWidget(self)
        self.main_widget = QWidget(self)

        # 새로운 패널을 위한 MainWidget
        self.additional_panel = MainWidget(self, self.main_widget)

        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.addWidget(self.video_widget)
        self.main_layout.addWidget(self.additional_panel)

        self.main_widget.setLayout(self.main_layout)

        self.stacked_widget.addWidget(self.main_widget)

        self.loading_widget.loading_complete.connect(self.show_video_widget)

        exit_action = QAction(QIcon('exit.png'), 'Exit', self)
        exit_action.setShortcut('Ctrl+W')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)  # Ensure that the native menu bar is not used
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(exit_action)

    def show_video_widget(self):
        self.stacked_widget.setCurrentWidget(self.main_widget)
        self.statusBar().showMessage('On Air')

    def toggle_gesture_mode(self, state):
        if state == Qt.Checked:
            self.statusBar().showMessage('Gesture Mode On')
        else:
            self.statusBar().showMessage('Gesture Mode Off')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
