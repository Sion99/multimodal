# voice/voice_recognition.py
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from transformers import pipeline
import torch
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)

previous_voice_label = ""
count = 0


def launch_fn(prob_threshold=0.8, chunk_length_s=2.0, stream_chunk_s=0.25):
    global count
    voice_result = ""
    print("Voice recognition is running...")
    wake_word = ["forward", "backward", "two", "stop", "up", "down", "right", "left", "follow"]
    sampling_rate = classifier.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    recommand_set = ['none']

    def check_and_print(set):
        global count
        if set[-1] != set[-2]:
            count = 0
            return True
        elif count >= 4:
            count = 0
            return True
        else:
            count += 1

    print("Listening for wake word...")

    for prediction in classifier(mic):
        prediction = prediction[0]
        print(prediction["label"])
        if prediction["label"] in wake_word:
            if prediction["score"] > prob_threshold:
                recommand_set.append(prediction["label"])
                if check_and_print(recommand_set):
                    print(prediction["label"])
                    voice_result = prediction["label"]

                    with open('./voice/voice_label.txt', 'w') as fp_voice:
                        fp_voice.write(str(time.time()) + ',' + voice_result)

                    break

    return voice_result
