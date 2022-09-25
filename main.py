import train
import cv2
from handsdetector import HandsDetector
import classifier
import torch


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    detector = HandsDetector()

    net = classifier.Classifier(train.CLASS_NUM, training=False)
    net.load_state_dict(torch.load('./models/model.pth'))
    # net.to(0)
    print(net.eval())

    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            print("[Detcetor] Ignoring empty camera frame.")
            continue
        result = detector.hand_detect(img)
        print(net(result).tolist())
        # time.sleep(20)


if __name__ == '__main__':
    main()