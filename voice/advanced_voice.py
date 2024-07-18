from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from transformers import pipeline
import torch
import time
from multiprocessing import Process

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 'audio-classification' 모델 로드 (MIT/ast-finetuned-speech-commands-v2 사용)
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)

previous_voice_label = ""
count = 0

# 실시간 음성 인식 함수 정의
def launch_fn(
        prob_threshold=0.8,
        chunk_length_s=2.0,
        stream_chunk_s=0.25
):
    voice_result = ""
    print("voice.py start!")
    # 감지할 명령어 리스트
    wake_word = ["click", "double click", "stop", "클릭", "더블클릭", "스탑"]
    sampling_rate = classifier.feature_extractor.sampling_rate

    # ffmpeg_microphone_live 함수를 통해 마이크로폰으로부터 실시간 오디오 스트림을 얻는다.
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

    # 각 단어에 대한 예측이 이루어질 때마다,
    # 예측된 라벨이 wake_word 중 하나인지, 그리고 해당 예측의 점수가 설정된 임계값(prob_threshold)보다 높은지 확인한다.
    for prediction in classifier(mic):
        prediction = prediction[0]
        print(prediction["label"])
        if prediction["label"] in wake_word:
            if prediction["score"] > prob_threshold:
                recommand_set.append(prediction["label"])
                if check_and_print(recommand_set):
                    print(f"Detected command: {prediction['label']}")
                    voice_result = prediction["label"]

                    # 두 조건이 모두 만족하면, 해당 라벨을 'voice_label.txt' 파일에 기록한다.
                    with open('./voiceAndgesture/voice_label.txt', 'w') as fp_voice:
                        fp_voice.write(f"{time.time()},{voice_result}")

    return voice_result

launch_fn()
