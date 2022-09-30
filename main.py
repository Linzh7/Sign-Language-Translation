import train
import cv2
from handsdetector import HandsDetector
import classifier
import torch
import labelmap
import mediapipe as mp
from queue import Queue
import playsound

SHOW = True

media_list = [
    "./media/good.mp3", "./media/bad.mp3", "./media/timeout.mp3",
    "./media/horns.mp3", "./media/victory.mp3"
]

THRESHOLD = -0.07  # log10(0.85)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    detector = HandsDetector()

    net = classifier.Classifier(train.CLASS_NUM, training=False)
    net.load_state_dict(torch.load('./models/model.pth'))
    # net.to(0)
    print(net.eval())

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    queue = Queue()
    for i in range(5):
        queue.put(i)

    cache = -1

    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            print("[Detcetor] Ignoring empty camera frame.")
            continue
        result = detector.hand_detect(img)
        if sum(result) != 0 and len(result) == 126:
            inference = net(result).tolist()
            inference_index = inference.index(max(inference))
            if inference[inference_index] > THRESHOLD:
                # print(inference[inference_index])
                print(labelmap.reversed_map[inference_index])
                queue.get()
                queue.put(inference_index)
                if max(queue.queue) == min(queue.queue):
                    if cache == queue.queue[0]:
                        continue
                    cache = queue.queue[0]
                    playsound.playsound(media_list[inference_index])
            if SHOW:
                for hand_landmarks in detector.landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                cv2.imshow('MediaPipe Hands', img)
                cv2.waitKey(1)


if __name__ == '__main__':
    main()