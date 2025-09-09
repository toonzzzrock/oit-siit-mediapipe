import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple

class HEAD_DIRECTION_TASK():
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 時間管理用変数
        self.off_start_time = None
        self.warning_active = False

    def get_head_direction(self, landmarks, img_w, img_h):
        # 鼻と左右目の位置を取得
        nose = landmarks[1]        # 鼻先
        left_eye = landmarks[33]   # 左目端
        right_eye = landmarks[263] # 右目端

        # ピクセル座標に変換
        nose_point = np.array([nose.x * img_w, nose.y * img_h])
        left_point = np.array([left_eye.x * img_w, left_eye.y * img_h])
        right_point = np.array([right_eye.x * img_w, right_eye.y * img_h])

        eye_center = (left_point + right_point) / 2
        dx = nose_point[0] - eye_center[0]
        dy = nose_point[1] - eye_center[1]

        # 閾値調整
        direction = "Front"
        if dx > 20:
            direction = "Looking Right"
        elif dx < -20:
            direction = "Looking Left"
        elif dy > 40:
            direction = "Looking Down"
        elif dy < -40:
            direction = "Looking Up"

        return direction

    def process(self, frame) -> Tuple[bool, str, np.ndarray]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        h, w, _ = frame.shape
        status = False
        message = "OK"  # 正常時は OK

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                direction = self.get_head_direction(face_landmarks.landmark, w, h)

                # 顔の枠描画
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Front 以外なら時間を測定
                if direction != "Front":
                    if self.off_start_time is None:
                        self.off_start_time = time.time()
                    else:
                        elapsed = time.time() - self.off_start_time
                        if elapsed >= 3:  # 3秒以上
                            self.warning_active = True
                else:
                    # 正面を向いたらリセット
                    self.off_start_time = None
                    self.warning_active = False

                # 警告表示
                if self.warning_active:
                    status = True
                    message = "WARNING: Head Direction!"
                    #cv2.putText(frame, message, (30, 100),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return status, message, frame