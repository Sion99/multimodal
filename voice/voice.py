from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from transformers import pipeline
import torch
import time
from multiprocessing import Process

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 우선, pipeline 함수를 이용해 'audio-classification' 모델을 로드하며,
# "MIT/ast-finetuned-speech-commands-v2"라는 사전 학습된 모델을 사용한다.
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)

previous_voice_label = ""
count = 0


# 이 모델은 특정 단어나 문장을 인식하도록 학습되었다.
# launch_fn 함수는 마이크로폰을 통해 실시간으로 입력되는 음성을 인식하고,
# 그 음성이 사전에 정의된 명령어(wake word)에 해당하는지 확인하는 역할을
def launch_fn(
        prob_threshold=0.8,
        chunk_length_s=2.0,
        stream_chunk_s=0.25
):
    voice_result = ""
    print("voice.py start!")
    wake_word = ["forward", "backward", "two", "stop", "up", "down", "right", "left", "follow"]
    sampling_rate = classifier.feature_extractor.sampling_rate

    # ffmpeg_microphone_live 함수를 통해 마이크로폰으로부터 실시간 오디오 스트림을 얻는다.
    # 이 스트림은 classifier 모델에 전달되어 음성 인식이 이루어진다.
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
        # print(prediction)
        # [{'score': 0.16547811031341553, 'label': 'follow'},
        # {'score': 0.09163922071456909, 'label': 'stop'},
        # {'score': 0.057492565363645554, 'label': 'off'},
        # {'score': 0.054507724940776825, 'label': 'down'},
        # {'score': 0.040247734636068344, 'label': 'seven'}]
        prediction = prediction[0]
        print(prediction["label"])
        if prediction["label"] in wake_word:
            if prediction["score"] > prob_threshold:
                recommand_set.append(prediction["label"])
                if check_and_print(recommand_set):
                    # print(recommand_set)
                    print(prediction["label"])
                    voice_result = prediction["label"]

                    # 만약 두 조건이 모두 만족하면, 해당 라벨을 'voice_label.txt' 파일에 기록한다.
                    fp_voice = open('./voiceAndgesture/voice_label.txt', 'w')
                    fp_voice.write(str(time.time()) + ',' + voice_result)
                    fp_voice.close()

                    # break

    return voice_result


launch_fn()

