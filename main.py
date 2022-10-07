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

media_list = []
for k, v in labelmap.label_map.items():
    media_list.append(f'./media/{k}.mp3')

THRESHOLD = -0.02  # log10(0.95)


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
                # print(list(map(lambda x: 10**x, inference)))
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
                (w, h), _ = cv2.getTextSize(
                    labelmap.reversed_map[inference_index],
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
                img = cv2.rectangle(img, (20, 0), (20 + w, 70),
                                    (255, 255, 255), -1)
                img = cv2.putText(img, labelmap.reversed_map[inference_index],
                                  (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                  (0, 0, 0), 5, cv2.LINE_AA)
                cv2.imshow('MediaPipe Hands', img)
                cv2.waitKey(1)


if __name__ == '__main__':
    main()