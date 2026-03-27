import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="day-2/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cam = cv2.VideoCapture(0)

canvas = None
prev_point = None
alpha = 0.7

sel_color = (0, 255, 0)

with vision.HandLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = cam.read()
        if not success:
            break

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = cv2.getTickCount() // (cv2.getTickFrequency() / 1000)
        result = detector.detect_for_video(mp_image, int(timestamp))

        h, w, _ = frame.shape
        pinch_detected = False

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                x1, y1 = int(hand[8].x * w), int(hand[8].y * h)  # index tip
                x2, y2 = int(hand[4].x * w), int(hand[4].y * h)  # thumb tip

                threshold = 25

                if (x2 - x1) ** 2 + (y2 - y1) ** 2 < threshold ** 2:
                    pinch_detected = True

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    current_point = (cx, cy)

                    # smooth current point
                    if prev_point is not None:
                        smooth_x = int(alpha * current_point[0] + (1 - alpha) * prev_point[0])
                        smooth_y = int(alpha * current_point[1] + (1 - alpha) * prev_point[1])
                        current_point = (smooth_x, smooth_y)

                        # draw connected line only during same pinch stroke
                        cv2.line(canvas, prev_point, current_point, sel_color, 3)

                    prev_point = current_point
                    break  # use first pinching hand only

        # IMPORTANT: when pinch stops, break the stroke
        if not pinch_detected:
            prev_point = None

        frame = cv2.add(frame, canvas)

        frame = cv2.putText(frame, "Press 'G' for Color 1", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Press 'H' for Color 2", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): 
            break
        elif key == ord("r"):
            canvas = np.zeros_like(frame)
            prev_point = None
        elif key == ord("g"):
            sel_color = (0, 255, 0)
        elif key == ord("h"):
            sel_color = (0, 0, 255)

cam.release()
cv2.destroyAllWindows()