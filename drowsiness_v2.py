"""
AI DRIVER DROWSINESS DETECTION SYSTEM
With Custom Danger Alarm (WAV)
macOS Compatible – afplay

Author: Your Name
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ================= CONFIG =================
EAR_THRESHOLD = 0.25
DROWSY_TIME_THRESHOLD = 2.5
SEVERE_TIME_THRESHOLD = 5.0
VOICE_ALERT_COOLDOWN = 5  # seconds

# ================= AUDIO FUNCTIONS =================
def play_alarm():
    # Force max volume and play alarm (non-blocking)
    os.system(
        'osascript -e "set volume output volume 100" && '
        'afplay danger_alarm.wav &'
    )

def stop_alarm():
    # Stop any playing alarm
    os.system('pkill afplay')

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# ================= CAMERA =================
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Camera not accessible")

# ================= LANDMARKS =================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ================= UTIL =================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ================= STATES =================
eye_closed_start_time = None
last_voice_alert_time = 0
driver_status = "SAFE"

# ================= MAIN LOOP =================
while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # ================= FACE NOT DETECTED =================
    if not results.multi_face_landmarks:
        driver_status = "EMERGENCY - DRIVER NOT DETECTED"
        play_alarm()

    else:
        landmarks = results.multi_face_landmarks[0].landmark

        # -------- FACE BOUNDING BOX --------
        xs = [int(p.x * w) for p in landmarks]
        ys = [int(p.y * h) for p in landmarks]
        cv2.rectangle(
            frame,
            (min(xs), min(ys)),
            (max(xs), max(ys)),
            (255, 0, 0),
            2
        )

        # -------- EYES --------
        left_eye = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE],
            dtype=np.int32
        )
        right_eye = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE],
            dtype=np.int32
        )

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear_avg = (ear_left + ear_right) / 2.0

        cv2.polylines(frame, [left_eye], True, (0, 255, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 255, 255), 2)

        # -------- DROWSINESS LOGIC --------
        if ear_avg < EAR_THRESHOLD:
            if eye_closed_start_time is None:
                eye_closed_start_time = current_time

            closed_time = current_time - eye_closed_start_time

            if closed_time >= SEVERE_TIME_THRESHOLD:
                driver_status = "EMERGENCY - DRIVER SLEEPING"
                play_alarm()

            elif closed_time >= DROWSY_TIME_THRESHOLD:
                driver_status = "DROWSY"
                stop_alarm()

                # Optional voice alert
                if current_time - last_voice_alert_time > VOICE_ALERT_COOLDOWN:
                    os.system('say "Alert. Driver is drowsy."')
                    last_voice_alert_time = current_time
            else:
                driver_status = "SAFE"
                stop_alarm()
        else:
            eye_closed_start_time = None
            driver_status = "SAFE"
            stop_alarm()

    # ================= UI =================
    color = (0, 255, 0)
    if "EMERGENCY" in driver_status:
        color = (0, 0, 255)
    elif "DROWSY" in driver_status:
        color = (0, 165, 255)

    cv2.putText(
        frame,
        f"STATUS : {driver_status}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.imshow("AI Driver Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
stop_alarm()
camera.release()
cv2.destroyAllWindows()
