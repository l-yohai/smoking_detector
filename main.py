# import the necessary packages
import RPi.GPIO as GPIO
from multiprocessing import Process
import os
import threading
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import time
# import pygame

# def play_music():
#    pygame.mixer.init(44100, -16, 1, 2048)
#    pygame.mixer.music.load('audio/warning.wav')
#    pygame.mixer.music.play()
#    while pygame.mixer.music.get_busy():
#          pygame.time.Clock().tick(10)
#    pygame.mixer.quit()


def detect_and_predict_smoking(frame, faceNet, smokeNet):
    global args

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = smokeNet.predict(faces, batch_size=32)

    return (locs, preds)


def start_smoking_detecting():
    global args

    # Face Detect 사전학습 모델 load
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # Smoking Detect 모델 load
    print("[INFO] loading smoking detector model...")
    smokeNet = load_model(args["model"])

    # Picam 비디오 스트리밍 시작
    print("[INFO] starting video stream...")
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    # 파이캠 무한루프 시작
    while True:
        # 비디오 스트리밍 사이즈 500으로 고정
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # 흡연 감지 모델을 통해 예측하고, 예측값 불러오기
        (locs, preds) = detect_and_predict_smoking(frame, faceNet, smokeNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (no_smoking, smoking) = pred

            # 흡연 및 비흡연 예측값들 저장
            no_smoking_preds = no_smoking
            smoking_preds = smoking

            # 카운트다운을 위한 시작시간 측정
            start = time.time()

            # 흡연을 하고 있다면, 흡연중이라는 메세지를, 아니라면 비흡연중이라는 메세지를 띄움.
            if no_smoking > smoking:
                label = "No smoking."
                color = (0, 255, 0)
            else:
                tmp = 0
                # 흡연 감지 시 3초동안 반복을 돌면서 예측값들 저장
                while (time.time() - start < 3):
                    no_smoking_preds += no_smoking
                    smoking_preds += smoking
                    tmp += 1

                # 디버깅을 위한 반복횟수, 비흡연 예측값들의 총합, 흡연 예측값들의 총합
                print("{} : {}, {}, {}".format(threading.currentThread(
                ).getName(), tmp, no_smoking_preds, smoking_preds))

                # 흡연 감지 3초 후의 예측값을 비교하고, 흡연 예측 값의 평균이 80% 이상이면 흡연중이라고 판단.
                if smoking_preds / tmp > 0.8:
                    label = "You're smoking!"
                    color = (0, 0, 255)
                    # Warning audio 재생
                    # play_music()

            # 스트리밍 프레임 박스에 메세지 띄우기
            cv2.putText(frame, label, (startX-50, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Smoking Detector", frame)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()
    vs.stop()


def gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(26, GPIO.IN)
    GPIO.setup(13, GPIO.IN)
    GPIO.setup(19, GPIO.IN)
    GPIO.setup(20, GPIO.IN)
    GPIO.setup(21, GPIO.IN)  # 센서 입력
    while True:
        if GPIO.input(19) and GPIO.input(20) and GPIO.input(26) and GPIO.input(21) and GPIO.input(13) == 1:
            print("{} is start : {}".format(
                threading.currentThread().getName(), 'Not Fire'))
            time.sleep(5)
        else:
            print("{} is start : {}".format(
                threading.currentThread().getName(), 'Fire detected'))
            time.sleep(5)


if __name__ == '__main__':
    # argument parsing 및 옵션 추가
    # -m 혹은 --model 로 custom model 추가 가능
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
                    default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
                    default="my.model",
                    help="path to trained smiking detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    th1 = threading.Thread(target=start_smoking_detecting)
    th2 = threading.Thread(target=gpio)

    th1.start()
    time.sleep(10)
    th2.start()
    th1.join()
    th2.join()
