import os
import cv2
import time
import numpy as np
from typing import Tuple
import mediapipe as mp

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# 目のランドマークID
LEFT_EYE_IDS  = [33, 159, 145, 133]
RIGHT_EYE_IDS = [362, 386, 374, 263]

def eye_aspect_ratio(landmarks, eye_ids, iw, ih):
    left  = np.array([landmarks[eye_ids[0]].x * iw, landmarks[eye_ids[0]].y * ih])
    top   = np.array([landmarks[eye_ids[1]].x * iw, landmarks[eye_ids[1]].y * ih])
    bottom= np.array([landmarks[eye_ids[2]].x * iw, landmarks[eye_ids[2]].y * ih])
    right = np.array([landmarks[eye_ids[3]].x * iw, landmarks[eye_ids[3]].y * ih])
    
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal

class IRIS_TASK():
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.EAR_THRESHOLD = 0.25
        self.CLOSE_TIME_THRESHOLD = 3.0
        self.last_eye_open_time = time.time()

    def process(self, frame) -> Tuple[bool, str, np.ndarray]:
        img = frame.copy()
        ih, iw, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        warning_flag = False
        message = "eyes open"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_ear  = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDS, iw, ih)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDS, iw, ih)
            avg_ear = (left_ear + right_ear) / 2.0

            # ランドマーク表示
            for idx in LEFT_EYE_IDS + RIGHT_EYE_IDS:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(img, (x, y), 2, (0, 255, 255), -1)

            if avg_ear < self.EAR_THRESHOLD:
                elapsed = time.time() - self.last_eye_open_time
                if elapsed >= self.CLOSE_TIME_THRESHOLD:
                    warning_flag = True
                    message = f"WARNING: eyes closed for {elapsed:.1f} sec"
            else:
                self.last_eye_open_time = time.time()
        else:
            self.last_eye_open_time = time.time()

        return warning_flag, message, img


class CELL_PHONE_TASK():
    def __init__(self):
        pass
    def process(self, frame) -> Tuple[bool, str, np.ndarray]:
        return True, "Cell phone detected", frame

class HEAD_DIRECTION_TASK():
    def __init__(self):
        pass
    def process(self, frame) -> Tuple[bool, str, np.ndarray]:
        return True, "side", frame

class HAND_TASK():
    def __init__(self):
        pass
    def process(self, frame) -> Tuple[bool, str, np.ndarray]:
        return True, "hand out of steering", frame