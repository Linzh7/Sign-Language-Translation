import mediapipe as mp
import cv2
from handsdetect import HandsDetector
import csv

label = 'flat'


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    detector = HandsDetector()

    with open(f'./dataset/{label}.csv', 'x') as file:
        writer = csv.writer(file)
        results_list = []
        while cap.isOpened():
            ok, img = cap.read()
            if not ok:
                print("[Detcetor] Ignoring empty camera frame.")
                continue
            result = detector.hand_detect(img)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if result:
                for hand_landmarks in detector.landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            img = cv2.putText(img, "'Ese' to exit, 'space' to capture",
                              (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('MediaPipe Hands', img)
            k = cv2.waitKey(33)
            if k == 27:
                break
            elif k == -1:
                continue
            elif k == 32:
                result += [label]
                results_list.append(result)
        writer.writerows(results_list)


if __name__ == '__main__':
    main()