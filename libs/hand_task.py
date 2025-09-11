import os
import numpy as np
from typing import Tuple
import mediapipe as mp
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

class HAND_TASK():
    def __init__(self, boundary = [0.2, 0.5, 0.8, 0.98]):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.boundary = boundary
        self.all_hand_center_points = []


    def process(self, frame) -> Tuple[bool, str, np.ndarray]:
        h, w, _  = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Define steering wheel region (adjust this based on your camera angle)
        roi_top_left = (int(w * self.boundary[0]), int(h * self.boundary[1]))
        roi_bottom_right = (int(w * self.boundary[2]), int(h * self.boundary[3]))
        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)


        hands_on_wheel = False
        self.all_hand_center_points = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                point = hand_landmarks.landmark[9]
                x, y = int(point.x * w), int(point.y * h)
                self.all_hand_center_points.append([x, y]) # the middle


            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)

                    # Check if landmark is inside steering wheel ROI
                    if (
                        (roi_top_left[0] <= x <= roi_bottom_right[0]) and
                        (roi_top_left[1] <= y <= roi_bottom_right[1])
                    ):
                        hands_on_wheel = True
                        break

        if hands_on_wheel:
            return False, "Hands on steering wheel", frame
        else:
            return True, "Warning: Hands removed from steering wheel!", frame
        
