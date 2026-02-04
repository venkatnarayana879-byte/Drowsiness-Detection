"""
AI DRIVER DROWSINESS MONITORING SYSTEM
CONTINUOUS DANGER ALARM (EXTERNAL SPEAKER READY)

Author: Your Name
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ================= CONFIGURATION =================
EAR_THRESHOLD = 0.25
DROWSY_TIME_SEC = 2.5
SEVERE_TIME_SEC = 5.0

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# ================= CAMERA =================
camera = None
for i in [0, 1, 2]:
    cam = cv2.VideoCapture(i)
    if cam.isOpened():
        camera = cam
        break

if camera is None:
    raise RuntimeError("Camera not accessible")

# ================= LANDMARKS =================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ================= FUNCTIONS =================
def EAR(eye):
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])
    h = np.linalg.norm(eye[0] - eye[3])
    return (v1 + v2) / (2.0 * h)

def danger_alarm(frame):
    # -------- RED FLASH (VISUAL DANGER) --------
    overlay = frame.copy()
    overlay[:] = (0, 0, 255)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # -------- FORCE MAX VOLUME + CONTINUOUS SIREN --------
    os.system(
        'osascript -e "set volume output volume 100" && '
        'afplay /System/Library/Sounds/Basso.aiff & '
        'afplay /System/Library/Sounds/Funk.aiff & '
        'afplay /System/Library/Sounds/Sosumi.aiff &'
    )

# ================= STATES =================
eye_closed_since = None
driver_state = "SAFE"

# ================= MAIN LOOP =================
while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # ============ DRIVER NOT DETECTED ============
    if not results.multi_face_landmarks:
        driver_state = "EMERGENCY - DRIVER NOT DETECTED"
        eye_closed_since = None
        danger_alarm(frame)

    else:
        lm = results.multi_face_landmarks[0].landmark

        # -------- FACE BOX --------
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]
        cv2.rectangle(
            frame,
            (min(xs), min(ys)),
            (max(xs), max(ys)),
            (255, 0, 0),
            2
        )

        # -------- EYES --------
        left_eye = np.array(
            [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE],
            dtype=np.int32
        )
        right_eye = np.array(
            [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE],
            dtype=np.int32
        )

        left_ear = EAR(left_eye)
        right_ear = EAR(right_eye)

        cv2.polylines(frame, [left_eye], True, (0, 255, 255), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 255), 1)

        # -------- DROWSINESS LOGIC --------
        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            if eye_closed_since is None:
                eye_closed_since = now

            closed_time = now - eye_closed_since

            if closed_time >= SEVERE_TIME_SEC:
                driver_state = "EMERGENCY - DRIVER SLEEPING"
                danger_alarm(frame)

            elif closed_time >= DROWSY_TIME_SEC:
                driver_state = "DROWSY"
            else:
                driver_state = "SAFE"
        else:
            eye_closed_since = None
            driver_state = "SAFE"

    # ================= UI =================
    color = (0, 255, 0)
    if "EMERGENCY" in driver_state:
        color = (0, 0, 255)
    elif "DROWSY" in driver_state:
        color = (0, 165, 255)

    cv2.putText(
        frame,
        f"STATUS : {driver_state}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.imshow("AI Driver Drowsiness Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
