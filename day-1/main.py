import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pygame
import time


pygame.mixer.init()
pygame.mixer.music.load('day-1/song.mp3')


# --- 1. MediaPipe Tasks Setup ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='day-1/face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO
)

# --- 2. Calibration Settings ---
# We use a deque to keep a "moving window" of the last 100 neutral frames
history_size = 100 
ratio_history = deque(maxlen=history_size)
base_ratio = 1.0

# --- 3. Main Loop ---
with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    print("System Starting... Look straight at the camera to calibrate.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Flip for mirror effect and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Required for Video Mode
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            
            # Key Landmark Indices:
            # 1: Nose Bridge (between eyes)
            # 4: Nose Tip
            # 152: Chin
            
            # Calculate Vertical Distances
            d_bridge = abs(landmarks[4].y - landmarks[1].y)
            d_chin = abs(landmarks[152].y - landmarks[4].y)
            
            # Avoid division by zero
            current_ratio = d_bridge / d_chin if d_chin != 0 else 1.0

            # --- 4. Continuous Auto-Calibration Logic ---
            ratio_history.append(current_ratio)
            
            # Calculate the average of our history to find the 'Neutral' baseline
            base_ratio = sum(ratio_history) / len(ratio_history)

            # --- 5. Detection Logic ---
            # If current ratio is significantly smaller than average -> UP
            # If current ratio is significantly larger than average -> DOWN
            sensitivity = 0.15
            lower_limit = base_ratio * (1 - sensitivity)
            upper_limit = base_ratio * (1 + sensitivity)

            if len(ratio_history) < history_size:
                status = f"Learning Face... {len(ratio_history)}%"
                color = (0, 165, 255) 
            else:
                if current_ratio < lower_limit:
                    status, color = "PAUSED (Tilt Up)", (0, 0, 255)
                    if pygame.mixer.music.get_busy(): pygame.mixer.music.pause()
                elif current_ratio > upper_limit:
                    status, color = "PLAYING (Tilt Down)", (0, 255, 0)
                    if not pygame.mixer.music.get_busy(): pygame.mixer.music.play()
                else:
                    status, color = "NEUTRAL", (255, 255, 255)

            # --- 6. Visuals ---
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, thick = 0.6, 1
            
            cv2.putText(frame, f"Current Ratio: {current_ratio:.3f}", (20, 40), font, fs, (255, 255, 255), thick)
            cv2.putText(frame, f"Base (Avg):    {base_ratio:.3f}", (20, 70), font, fs, (200, 200, 200), thick)
            cv2.putText(frame, f"Lower Thr:     {lower_limit:.3f}", (20, 100), font, fs, (0, 0, 255), thick)
            cv2.putText(frame, f"Upper Thr:     {upper_limit:.3f}", (20, 130), font, fs, (0, 255, 0), thick)
            
            # Big status text
            cv2.putText(frame, status, (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        cv2.imshow('Adaptive Head Tilt Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()