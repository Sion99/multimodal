import speech_recognition as sr
import ssl

r = sr.Recognizer()

ssl._create_default_https_context = ssl._create_unverified_context

with sr.Microphone() as source:
    print('listening...')
    audio = r.listen(source, timeout=10, phrase_time_limit=10)
    print('.....')

try:
    text = r.recognize_whisper(audio, language='ko')
    print(text)
except sr.UnknownValueError:
    print('Recognizer Failed..')
except sr.RequestError as e:
    print('Request Failed...', e)
