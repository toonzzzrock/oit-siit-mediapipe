import os, time, urllib.request
from pathlib import Path
import numpy as np
from typing import Tuple
import mediapipe as mp
import cv2

class CELL_PHONE_TASK():
    base_url = 'https://storage.googleapis.com/mediapipe-tasks/object_detector_without_nms/'
    model_name = 'efficientdet_lite0_int8_no_nms.tflite'
    model_folder_path = r'.\learned_models\mediapipe'  # may be relative at source, we resolve it

    def __init__(self,
                 model_folder_path=model_folder_path,
                 base_url=base_url,
                 model_name=model_name,
                 max_results=2,
                 score_threshold=0.3,
                 mode="vedio"):
        self.mode = mode
        self.score_threshold = score_threshold
        rmode = mp.tasks.vision.RunningMode.IMAGE if mode=="image" else mp.tasks.vision.RunningMode.VIDEO

        # 1) Resolve to ABSOLUTE path (prevents site-packages prefixing your path)
        model_path = Path(self._ensure_model(base_url, model_folder_path, model_name)).resolve()
        # model_path = r"C:\oit\py25en\source\projects\learned_models\mediapipe\efficientdet_lite0_fp32.tflite"
        buf = model_path.read_bytes()

        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=buf),
            max_results=max_results,
            score_threshold=score_threshold,
            running_mode=rmode
        )
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 2) Monotonic timestamp for VIDEO mode (strictly increasing)
        self._t0 = time.perf_counter()

    def _ensure_model(self, base_url, folder, filename) -> Path:
        model_dir = Path(folder).expanduser().resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        if not model_path.exists():
            url = base_url + filename
            urllib.request.urlretrieve(url, model_path.as_posix())
        return model_path

    def process(self, frame) -> Tuple[bool, str, np.ndarray]:
        detected, message = False, "No object detected in hand"

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # hand_results = self.hands.process(img_rgb)

        # if hand_results.multi_hand_landmarks:
        #     for hand_landmarks in hand_results.multi_hand_landmarks:
                # # Draw hand landmarks (optional)
                # self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        if True:
            # Detect on the full frame (can switch to cropped ROI if desired)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts_ms = int((time.perf_counter() - self._t0) * 1000)  # strictly increasing
            detection_result = self.detector.detect_for_video(mp_image, ts_ms)

            for det in detection_result.detections:
                category = det.categories[0].category_name or "object"
                score = det.categories[0].score
                bbox = det.bounding_box
                x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
                
                if (
                    (category.lower() not in {"cell phone", "cellphone", "mobile phone", "phone"}) or
                    (score < self.score_threshold)
                ):
                    continue

                detected = True
                message = f"{category} ({score:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, message, (x1, max(0, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        return detected, message, frame
