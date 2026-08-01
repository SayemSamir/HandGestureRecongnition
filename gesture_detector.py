import os
import time
import urllib.request
from collections import Counter, deque

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

GESTURE_MESSAGES = {
    "Thumbs Up": "Great!",
    "Peace": "Peace",
    "OK": "OK",
    "Open Palm": "Stop",
    "None": "Show a gesture...",
}

GESTURE_COLORS = {
    "Thumbs Up": (0, 200, 0),
    "Peace": (255, 191, 0),
    "OK": (0, 165, 255),
    "Open Palm": (60, 60, 240),
    "None": (180, 180, 180),
}


def _distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def ensure_model(model_path=MODEL_PATH, model_url=MODEL_URL):
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path

    print("Downloading hand landmark model (~7.8 MB, one-time)...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except Exception as exc:
        raise RuntimeError(f"couldn't download model from {model_url}: {exc}") from exc

    return model_path


class GestureDetector:
    def __init__(self, detection_confidence=0.6, presence_confidence=0.6, tracking_confidence=0.6,
                 smoothing_window=8, model_path=None):
        model_path = model_path or ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start_time = time.time()

        self.gesture_history = deque(maxlen=smoothing_window)
        self.last_stable_gesture = "None"

    def close(self):
        self.landmarker.close()

    @staticmethod
    def bounding_box(landmarks_px, frame_width, frame_height, padding=25):
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        x1, x2 = max(min(xs) - padding, 0), min(max(xs) + padding, frame_width)
        y1, y2 = max(min(ys) - padding, 0), min(max(ys) + padding, frame_height)
        return x1, y1, x2, y2

    @staticmethod
    def fingers_up(lm):
        fingers = [0, 0, 0, 0, 0]

        hand_scale = _distance(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6
        thumb_extension = _distance(lm[THUMB_TIP], lm[INDEX_MCP]) / hand_scale
        fingers[0] = 1 if thumb_extension > 0.4 else 0

        tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
        for i in range(4):
            fingers[i + 1] = 1 if lm[tips[i]][1] < lm[pips[i]][1] else 0

        return fingers

    def classify_gesture(self, lm, fingers):
        thumb, index, middle, ring, pinky = fingers
        hand_scale = _distance(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6
        thumb_index_dist = _distance(lm[THUMB_TIP], lm[INDEX_TIP]) / hand_scale

        if index and middle and ring and pinky:
            return "Open Palm"

        if index and middle and not ring and not pinky:
            return "Peace"

        if thumb_index_dist < 0.5 and middle and ring and pinky:
            return "OK"

        if thumb and not index and not middle and not ring and not pinky:
            margin = hand_scale * 0.15
            if lm[THUMB_TIP][1] < lm[INDEX_MCP][1] - margin:
                return "Thumbs Up"

        return "Unknown"

    def smooth_gesture(self, raw_gesture):
        self.gesture_history.append(raw_gesture)
        counts = Counter(self.gesture_history)
        top_gesture, top_count = counts.most_common(1)[0]

        if top_count >= len(self.gesture_history) // 2 + 1:
            self.last_stable_gesture = top_gesture

        return self.last_stable_gesture

    def process_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            gesture = self.smooth_gesture("None")
            return {
                "hand_found": False,
                "gesture": gesture,
                "message": GESTURE_MESSAGES.get(gesture, GESTURE_MESSAGES["None"]),
                "color": GESTURE_COLORS.get(gesture, GESTURE_COLORS["None"]),
                "bbox": None,
                "landmarks_px": None,
                "debug": None,
            }

        landmarks = result.hand_landmarks[0]
        lm_px = [(int(p.x * w), int(p.y * h)) for p in landmarks]

        fingers = self.fingers_up(lm_px)
        raw_gesture = self.classify_gesture(lm_px, fingers)
        gesture = self.smooth_gesture(raw_gesture)

        return {
            "hand_found": True,
            "gesture": gesture,
            "message": GESTURE_MESSAGES.get(gesture, GESTURE_MESSAGES["None"]),
            "color": GESTURE_COLORS.get(gesture, GESTURE_COLORS["None"]),
            "bbox": self.bounding_box(lm_px, w, h),
            "landmarks_px": lm_px,
            "debug": {
                "fingers": fingers,
                "raw_gesture": raw_gesture,
            },
        }
