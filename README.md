# Drowsiness Detection System

A real-time driver drowsiness detection system that alerts when a driver shows signs of fatigue — built to address road safety.

---

## How it works

The system uses Eye Aspect Ratio (EAR) thresholding on MediaPipe's 468-point facial landmark mesh to detect eye closure patterns in real time. When the EAR drops below a threshold for a set number of frames, an alert is triggered.

- ~88% detection accuracy
- Runs at 30 FPS in real time
- Instant audio/visual alert on drowsiness detection

---

## Tech stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| OpenCV | Video capture and frame processing |
| MediaPipe | 468-point facial landmark detection |
| NumPy | EAR calculation |

---

## Getting started

```bash
git clone https://github.com/venkatnarayana879-byte/Drowsiness-Detection
cd Drowsiness-Detection
pip install -r requirements.txt
python detect.py
```

Make sure your webcam is connected before running.

---

## How EAR works

EAR (Eye Aspect Ratio) is calculated using 6 facial landmarks around each eye:
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

When both eyes are open, EAR stays above ~0.25. When closed or drowsy, it drops — triggering the alert.

---

## Results

| Metric | Value |
|--------|-------|
| Detection accuracy | ~88% |
| Processing speed | 30 FPS |
| Landmark points used | 468 |

---

## Built by

Velpuri Venkat Narayana — [Portfolio](https://venkat-aiml-portfolio.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/venkat-narayana-velpuri-7a08a0282/)
