import sys
import time

import cv2

from gesture_detector import HAND_CONNECTIONS, GestureDetector

WINDOW_NAME = "Hand Gesture Recognition"
CAM_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

PANEL_COLOR = (20, 20, 20)
ACCENT_COLOR = (0, 215, 255)


def draw_translucent_rect(frame, top_left, bottom_right, color, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_landmarks(frame, landmarks_px, connection_color=(255, 255, 255), point_color=(0, 215, 255)):
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_px[start], landmarks_px[end], connection_color, 2, cv2.LINE_AA)
    for x, y in landmarks_px:
        cv2.circle(frame, (x, y), 4, point_color, -1, cv2.LINE_AA)


def draw_bounding_box(frame, bbox, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    corner_len = 18
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, 4)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, 4)


def draw_hud(frame, result, fps):
    h, w = frame.shape[:2]

    draw_translucent_rect(frame, (0, 0), (330, 55), PANEL_COLOR, alpha=0.6)
    gesture_text = result["gesture"] if result["hand_found"] else "No hand detected"
    cv2.putText(frame, gesture_text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                result["color"], 2, cv2.LINE_AA)

    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    draw_translucent_rect(frame, (w - tw - 32, 0), (w, 45), PANEL_COLOR, alpha=0.6)
    cv2.putText(frame, fps_text, (w - tw - 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                ACCENT_COLOR, 2, cv2.LINE_AA)

    message = result["message"]
    font_scale = 1.4
    thickness = 3
    (mw, mh), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bar_top = h - mh - 50
    draw_translucent_rect(frame, (0, bar_top), (w, h), PANEL_COLOR, alpha=0.65)
    cv2.putText(frame, message, ((w - mw) // 2, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, result["color"], thickness, cv2.LINE_AA)

    cv2.putText(frame, "Press Q to quit, D for debug", (16, h - mh - 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (150, 150, 150), 1, cv2.LINE_AA)


def draw_debug(frame, result):
    debug = result.get("debug")
    if debug is None:
        lines = ["debug: no hand detected"]
    else:
        thumb, index, middle, ring, pinky = debug["fingers"]
        lines = [
            f"fingers T:{thumb} I:{index} M:{middle} R:{ring} P:{pinky}",
            f"raw gesture: {debug['raw_gesture']}",
        ]

    y = 75
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        draw_translucent_rect(frame, (10, y - th - 6), (10 + tw + 12, y + 6), PANEL_COLOR, alpha=0.6)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += th + 16


def open_webcam(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def main():
    cap = open_webcam(CAM_INDEX)
    if cap is None:
        print(f"Error: could not open webcam (index {CAM_INDEX}). "
              "Check that a camera is connected and not in use by another app.")
        sys.exit(1)

    try:
        detector = GestureDetector()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        cap.release()
        sys.exit(1)

    prev_time = time.time()
    fps_avg = 0.0
    debug_mode = False

    print("Hand Gesture Recognition running. Press Q to quit, D to toggle debug overlay.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: failed to read a frame from the webcam. Retrying...")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)

            result = detector.process_frame(frame)

            if result["hand_found"] and result["landmarks_px"] is not None:
                draw_landmarks(frame, result["landmarks_px"])
                if result["bbox"] is not None:
                    draw_bounding_box(frame, result["bbox"], result["color"])

            now = time.time()
            instant_fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            fps_avg = fps_avg * 0.9 + instant_fps * 0.1 if fps_avg else instant_fps
            prev_time = now

            draw_hud(frame, result, fps_avg)
            if debug_mode:
                draw_debug(frame, result)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("d"), ord("D")):
                debug_mode = not debug_mode

    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
