import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import platform

class WindowsAudio:
    def __init__(self):
        from pycaw.pycaw import AudioUtilities
        self.device = AudioUtilities.GetSpeakers()
        self.volume = self.device.EndpointVolume
    def get_current(self):
        current_scalar = self.volume.GetMasterVolumeLevelScalar()   # 0.0 to 1.0
        return int(current_scalar * 100)
    def set_percent(self, percent):
        percent = max(0, min(100, percent))
        self.volume.SetMasterVolumeLevelScalar(percent / 100, None)

class MacAudio:
    def __init__(self):
        import subprocess
        self.subprocess = subprocess

    def get_current(self):
        result = self.subprocess.check_output([
            "osascript",
            "-e",
            "output volume of (get volume settings)"
        ])
        return int(result.decode().strip())

    def set_percent(self, percent):
        percent = max(0, min(100, percent))
        self.subprocess.run([
            "osascript",
            "-e",
            f"set volume output volume {percent}"
        ], check=True)
    
class LinuxAudio:
    def __init__(self):
        import pulsectl
        self.pulse = pulsectl.Pulse('hand-volume-control')

    def get_default_sink(self):
        server = self.pulse.server_info()
        for sink in self.pulse.sink_list():
            if sink.name == server.default_sink_name:
                return sink
        raise RuntimeError("Default sink not found")

    def get_current(self):
        sink = self.get_default_sink()
        return round(sink.volume.value_flat * 100)

    def set_percent(self, percent):
        percent = max(0, min(100, percent))
        sink = self.get_default_sink()
        self.pulse.volume_set_all_chans(sink, percent / 100)

system = platform.system()
if system == "Windows":
    backend = WindowsAudio()
elif system == "Linux":
    backend = LinuxAudio()
elif system == "Darwin":
    backend = MacAudio()
else:
    raise RuntimeError("Unsupported OS")


last_change_time = 0
COOLDOWN = 0.3  # seconds (150ms)
MOVE_THRESHOLD = 12
VOLUME_STEP = 5

bar_x = 30
bar_y = 40
bar_w = 250
bar_h = 25


def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def can_change_volume(last_change_time, cooldown):
    return time.time() - last_change_time > cooldown

BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="day-3/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cam = cv2.VideoCapture(0)

prev_y = None

with vision.HandLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = cam.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = cv2.getTickCount() // (cv2.getTickFrequency() / 1000)
        result = detector.detect_for_video(mp_image, int(timestamp))
        h, w, _ = frame.shape

        is_Fist = False
        cur_y = None
        cur_percent = backend.get_current()

        filled_w = int((cur_percent / 100) * bar_w)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                lm = hand
                wrist = lm[0]
                cv2.circle(frame, (int(wrist.x * w), int(wrist.y * h)), 8, (0, 255, 255), -1)
                folded = 0
                # check 4 fingers (ignore thumb for stability)
                for tip_id in [8, 12, 16, 20]:
                    if dist(lm[tip_id], wrist) < 0.25:
                        folded += 1

                if folded >= 3:
                    is_Fist = True
                    cur_y = int(wrist.y * h)

        if is_Fist:
            if prev_y is None:
                prev_y = cur_y
            else:
                dy = cur_y - prev_y
                if dy < -MOVE_THRESHOLD:
                    # if can_change_volume(last_change_time, COOLDOWN):
                        backend.set_percent(cur_percent+VOLUME_STEP)
                        # last_change_time=time.time()
                elif dy > MOVE_THRESHOLD:
                    # if can_change_volume(last_change_time, COOLDOWN):
                        backend.set_percent(cur_percent-VOLUME_STEP)
                        # last_change_time=time.time()
                prev_y = cur_y
        else:
            prev_y = None

        
        # background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)

        # filled part
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

        # border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

        # text
        cv2.putText(frame, f"{cur_percent}%", (bar_x, bar_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)

        

cam.release()
cv2.destroyAllWindows()