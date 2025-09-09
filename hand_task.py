import os
import numpy as np
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
import winsound
import mediapipe as mp
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import time
import pathlib

class HAND_TASK():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils


    def process(self, frame) -> Tuple[bool, str, np.ndarray]:
        h, w, _,  = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Define steering wheel region (adjust this based on your camera angle)
        roi_top_left = (int(w * 0.2), int(h * 0.5))
        roi_bottom_right = (int(w * 0.8), int(h * 0.98))
        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

        hands_on_wheel = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)

                    # Check if landmark is inside steering wheel ROI
                    if roi_top_left[0] <= x <= roi_bottom_right[0] and roi_top_left[1] <= y <= roi_bottom_right[1]:
                        hands_on_wheel = True
                        break

        if hands_on_wheel:
            return False, "Hands on steering wheel", frame
        else:
            return True, "Warning: Hands removed from steering wheel!", frame
        
