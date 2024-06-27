# voice/voice_command.py
import time
from voice.voice_recognition import launch_fn
from multiprocessing import Process


def start_voice_recognition():
    process = Process(target=launch_fn)
    process.start()
    return process


def get_voice_cmd():
    voice_label = ''
    prev_timestamp = ''
    with open('./voice/voice_label.txt', 'r') as fp_voice:
        text = fp_voice.read()
        if text != '':
            timestamp, voice_label = text.split(',')
            if prev_timestamp != timestamp:
                prev_timestamp = timestamp
                print(prev_timestamp, voice_label)
    time.sleep(0.05)
    return voice_label


def reset_voice_label():
    with open('./voice/voice_label.txt', 'w') as fp_voice:
        fp_voice.write(str(time.time()) + ', -')
