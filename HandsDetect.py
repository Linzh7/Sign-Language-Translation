import mediapipe as mp
import numpy as np


class HandsDetector():
    def __init__(self):
        print('[Detcetor] init.')
        self.detector = mp.solutions.hands.Hands(static_image_mode=False,
                                                 model_complexity=0,
                                                 max_num_hands=2,
                                                 min_detection_confidence=0.5,
                                                 min_tracking_confidence=0.5)

    def hand_detect(self, img):
        print('[Detcetor] Start processing an image.', end='')
        img.flags.writeable = False
        hands_landmarks = self.detector.process(img).multi_hand_landmarks
        two_hands_landmarks = []
        if hands_landmarks != None:
            for one_hand_landmarks in hands_landmarks:
                for _, info in enumerate(one_hand_landmarks.landmark):
                    two_hands_landmarks.append([info.x, info.y, info.z])
                # landmarks = np.array(landmarks)
                # centered_landmarks.append(
                #     (landmarks - np.mean(landmarks, axis=0)).tolist())
            if len(hands_landmarks) == 1:
                two_hands_landmarks += [[0, 0, 0]] * 21
            landmarks = np.array(two_hands_landmarks)
            if landmarks.shape == (42, 3):
                centered_landmarks = ((landmarks - np.mean(landmarks, axis=0)))
                rescaled_landmarks = (centered_landmarks *
                                      (1 / centered_landmarks.max())).tolist()
                print(' Done')
                return rescaled_landmarks
        else:
            print(' No hand detected.')
        # print(centered_landmarks)


if __name__ == '__main__':
    import cv2
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    detector = HandsDetector()

    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            print("[Detcetor] Ignoring empty camera frame.")
            continue
        detector.hand_detect(img)