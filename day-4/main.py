import cv2
import mediapipe as mp
import time
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# BUILDCORED ORCAS — Day 04: BlinkLock (FaceLandmarker Tasks API)
# Rapid 3 blinks -> LOCK
# Slow deliberate wink -> UNLOCK
# PIN fallback included
# ============================================================

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "day-4/face_landmarker.task"   # change if needed

EAR_THRESHOLD = 0.25
BLINK_TIME_WINDOW = 2.0
BLINKS_TO_LOCK = 3
MIN_BLINK_FRAMES = 2
BLINK_COOLDOWN = 0.12

WINK_MIN_DURATION = 0.45
WINK_MAX_DURATION = 1.50

PIN_CODE = "1234"

# -----------------------------
# CAMERA SETUP
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

# -----------------------------
# MEDIAPIPE TASKS SETUP
# -----------------------------
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

# -----------------------------
# FACIAL LANDMARK INDICES
# -----------------------------
# Using the same landmark ids as your old FaceMesh version
LEFT_EYE_TOP = [159, 160, 161]
LEFT_EYE_BOTTOM = [145, 144, 153]
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = [386, 387, 388]
RIGHT_EYE_BOTTOM = [374, 373, 380]
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263


def get_ear(landmarks, top_ids, bottom_ids, left_id, right_id):
    vertical = 0.0
    for t, b in zip(top_ids, bottom_ids):
        vertical += abs(landmarks[t].y - landmarks[b].y)
    vertical /= len(top_ids)

    horizontal = abs(landmarks[left_id].x - landmarks[right_id].x)
    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def reset_unlock_trackers():
    return 0, 0


# -----------------------------
# STATES
# -----------------------------
STATE_IDLE = "IDLE"
STATE_COUNTING = "COUNTING"
STATE_LOCKED = "LOCKED"
STATE_PIN_ENTRY = "PIN_ENTRY"

state = STATE_IDLE

# -----------------------------
# BLINK STATE
# -----------------------------
blink_count = 0
counting_start_time = 0.0
both_closed_frames = 0
last_blink_time = 0.0

# -----------------------------
# WINK STATE
# -----------------------------
left_wink_frames = 0
right_wink_frames = 0
wink_message = ""

# -----------------------------
# PIN STATE
# -----------------------------
pin_buffer = ""
pin_message = "Press P for PIN"

print("\nBlinkLock is running!")
print(f"Model: {MODEL_PATH}")
print(f"EAR threshold: {EAR_THRESHOLD}")
print(f"Blink {BLINKS_TO_LOCK}x within {BLINK_TIME_WINDOW}s to LOCK")
print("Slow deliberate wink to UNLOCK")
print(f"Press digits for PIN ({PIN_CODE}) after pressing 'p'")
print("Press 'q' to quit.\n")

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Monotonically increasing timestamp in ms
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_ear = 0.0
        left_ear = 0.0
        right_ear = 0.0
        landmarks = None

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = get_ear(
                landmarks,
                LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                LEFT_EYE_LEFT, LEFT_EYE_RIGHT
            )
            right_ear = get_ear(
                landmarks,
                RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT
            )
            current_ear = (left_ear + right_ear) / 2

        # ============================================================
        # LOCKED / PIN MODE
        # ============================================================
        if state in (STATE_LOCKED, STATE_PIN_ENTRY):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

            cv2.putText(frame, "LOCKED", (w // 2 - 120, h // 2 - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

            cv2.putText(frame, "Unlock by slow wink or PIN",
                        (w // 2 - 190, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)

            cv2.putText(frame, "Press P for PIN entry",
                        (w // 2 - 135, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

            if landmarks is not None:
                left_closed = left_ear < EAR_THRESHOLD
                right_closed = right_ear < EAR_THRESHOLD

                # left wink
                if left_closed and not right_closed:
                    left_wink_frames += 1
                else:
                    if left_wink_frames > 0:
                        wink_seconds = left_wink_frames / 30.0
                        if WINK_MIN_DURATION <= wink_seconds <= WINK_MAX_DURATION:
                            state = STATE_IDLE
                            blink_count = 0
                            pin_buffer = ""
                            pin_message = "Unlocked by wink"
                            wink_message = "Unlocked by LEFT wink"
                            left_wink_frames, right_wink_frames = reset_unlock_trackers()
                            print("UNLOCKED by LEFT wink")
                    left_wink_frames = 0

                # right wink
                if right_closed and not left_closed:
                    right_wink_frames += 1
                else:
                    if right_wink_frames > 0:
                        wink_seconds = right_wink_frames / 30.0
                        if WINK_MIN_DURATION <= wink_seconds <= WINK_MAX_DURATION:
                            state = STATE_IDLE
                            blink_count = 0
                            pin_buffer = ""
                            pin_message = "Unlocked by wink"
                            wink_message = "Unlocked by RIGHT wink"
                            left_wink_frames, right_wink_frames = reset_unlock_trackers()
                            print("UNLOCKED by RIGHT wink")
                    right_wink_frames = 0

                cv2.putText(frame, f"Left EAR:  {left_ear:.3f}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            if state == STATE_PIN_ENTRY:
                cv2.putText(frame, "PIN ENTRY",
                            (w // 2 - 90, h // 2 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                cv2.putText(frame, "*" * len(pin_buffer),
                            (w // 2 - 20, h // 2 + 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, pin_message,
                            (w // 2 - 170, h // 2 + 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
            else:
                cv2.putText(frame, "Slow wink to unlock",
                            (w // 2 - 120, h // 2 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)
                cv2.putText(frame, pin_message,
                            (w // 2 - 110, h // 2 + 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

            cv2.putText(frame, "q = quit",
                        (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

            cv2.imshow("BlinkLock - Day 04", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                state = STATE_PIN_ENTRY
                pin_buffer = ""
                pin_message = "Type 4-digit PIN"
                print("PIN entry opened")
            elif state == STATE_PIN_ENTRY:
                if ord('0') <= key <= ord('9'):
                    pin_buffer += chr(key)
                    if len(pin_buffer) == len(PIN_CODE):
                        if pin_buffer == PIN_CODE:
                            state = STATE_IDLE
                            blink_count = 0
                            pin_message = "Unlocked by PIN"
                            wink_message = ""
                            left_wink_frames, right_wink_frames = reset_unlock_trackers()
                            print("UNLOCKED by PIN")
                        else:
                            pin_message = "Wrong PIN"
                            pin_buffer = ""
                elif key in (8, 127):
                    pin_buffer = pin_buffer[:-1]
                elif key == 27:
                    state = STATE_LOCKED
                    pin_buffer = ""
                    pin_message = "Press P for PIN"

            continue

        # ============================================================
        # NORMAL MODE
        # ============================================================
        if landmarks is not None:
            left_closed = left_ear < EAR_THRESHOLD
            right_closed = right_ear < EAR_THRESHOLD
            both_closed = left_closed and right_closed

            if both_closed:
                both_closed_frames += 1
            else:
                if both_closed_frames >= MIN_BLINK_FRAMES:
                    now = time.time()

                    if now - last_blink_time > BLINK_COOLDOWN:
                        last_blink_time = now

                        if state == STATE_IDLE:
                            state = STATE_COUNTING
                            blink_count = 1
                            counting_start_time = now
                            print(f"Blink {blink_count}/{BLINKS_TO_LOCK}")

                        elif state == STATE_COUNTING:
                            blink_count += 1
                            print(f"Blink {blink_count}/{BLINKS_TO_LOCK}")

                            if blink_count >= BLINKS_TO_LOCK:
                                state = STATE_LOCKED
                                blink_count = 0
                                pin_buffer = ""
                                pin_message = "Press P for PIN"
                                wink_message = ""
                                left_wink_frames, right_wink_frames = reset_unlock_trackers()
                                print("LOCKED")

                both_closed_frames = 0

            if state == STATE_COUNTING:
                elapsed = time.time() - counting_start_time
                if elapsed > BLINK_TIME_WINDOW:
                    print(f"Timeout. Only got {blink_count}/{BLINKS_TO_LOCK}. Resetting.")
                    state = STATE_IDLE
                    blink_count = 0

            for idx in LEFT_EYE_TOP + LEFT_EYE_BOTTOM + RIGHT_EYE_TOP + RIGHT_EYE_BOTTOM:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # ============================================================
        # DISPLAY
        # ============================================================
        ear_color = (0, 0, 255) if current_ear < EAR_THRESHOLD else (0, 255, 0)

        cv2.putText(frame, f"EAR: {current_ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        cv2.putText(frame, f"Left EAR: {left_ear:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        state_colors = {
            STATE_IDLE: (200, 200, 200),
            STATE_COUNTING: (0, 200, 255),
            STATE_LOCKED: (0, 0, 255),
            STATE_PIN_ENTRY: (180, 100, 255),
        }

        cv2.putText(frame, f"State: {state}", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_colors[state], 2)

        if state == STATE_COUNTING:
            elapsed = time.time() - counting_start_time
            remaining = max(0, BLINK_TIME_WINDOW - elapsed)

            cv2.putText(frame, f"Blinks: {blink_count}/{BLINKS_TO_LOCK}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f"Time left: {remaining:.1f}s", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            bar_width = int((remaining / BLINK_TIME_WINDOW) * 220)
            cv2.rectangle(frame, (10, 225), (10 + bar_width, 238), (0, 200, 255), -1)
            cv2.rectangle(frame, (10, 225), (230, 238), (100, 100, 100), 1)

        if wink_message:
            cv2.putText(frame, wink_message, (10, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)

        cv2.putText(
            frame,
            "3 rapid full blinks = LOCK | slow wink = UNLOCK | P = PIN | q = quit",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1
        )

        cv2.imshow("BlinkLock - Day 04", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\nBlinkLock ended.")