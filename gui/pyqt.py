import sys
import pyaudio
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QProgressBar, QLabel
from PyQt5.QtCore import QTimer, Qt


class AudioWidget(QWidget):
    def __init__(self, parent=None):
        super(AudioWidget, self).__init__(parent)

        # QLabel to display volume
        self.volume_label = QLabel('Volume Level', self)
        self.volume_label.setAlignment(Qt.AlignCenter)

        # QProgressBar to display volume level
        self.volume_bar = QProgressBar(self)
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setTextVisible(True)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.volume_label)
        layout.addWidget(self.volume_bar)
        self.setLayout(layout)

        # PyAudio
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
        self.timer_audio = QTimer(self)
        self.timer_audio.timeout.connect(self.update_audio)
        self.timer_audio.start(30)

    def update_audio(self):
        try:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            data_int = np.frombuffer(data, dtype=np.int16)
            volume_level = np.sqrt(np.mean(data_int ** 2))  # RMS calculation
            if np.isnan(volume_level):
                volume_level = 0
            volume_percentage = int((volume_level / 32768) * 100)
            self.volume_bar.setValue(volume_percentage)
            self.volume_label.setText(f'Volume Level: {volume_percentage}%')
        except IOError:
            pass

    def closeEvent(self, event):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Audio Input Visualization')
        self.setGeometry(100, 100, 400, 200)

        self.widget = AudioWidget(self)
        self.setCentralWidget(self.widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
