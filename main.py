import time
import cv2
from HandsDetect import HandsDetector


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    detector = HandsDetector()

    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            print("[Detcetor] Ignoring empty camera frame.")
            continue
        result = detector.hand_detect(img)

        time.sleep(20)


if __name__ == '__main__':
    main()