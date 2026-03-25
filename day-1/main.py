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
            
            if len(ratio_history) < history_size:
                status = f"Learning Face... {int(len(ratio_history))}%"
                color = (0, 165, 255) # Orange
            else:
                # Sensitivity thresholds (0.15 = 15% deviation)
                if current_ratio < base_ratio * 0.85:
                    status, color = "Music is paused", (0, 0, 255)
                    pygame.mixer.music.pause()
                elif current_ratio > base_ratio * 1.15:
                    status, color = "Playing music", (0, 255, 0)
                    pygame.mixer.music.play()
                else:
                    status, color = " ", (255, 255, 255)

            # --- 6. Visuals ---
            cv2.putText(frame, status, (30, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            
            # Optional: Draw a line showing the 'nose-to-chin' axis for debugging
            h, w, _ = frame.shape
            p_bridge = (int(landmarks[1].x * w), int(landmarks[1].y * h))
            p_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
            p_chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))
            
            cv2.line(frame, p_bridge, p_tip, (255, 0, 0), 2)
            cv2.line(frame, p_tip, p_chin, (0, 255, 255), 2)

        cv2.imshow('Adaptive Head Tilt Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()