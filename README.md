# 🚁 Drone Navigation using Real-Time Image Classification

This project implements a real-time drone vision system using deep learning to classify surroundings and make navigation decisions based on visual input. The drone detects scenes like **Forest, Fire, City, Animal, Vehicle, and Water** and performs specific navigation responses.

---

## 📦 Features

- ✅ Real-time video stream processing
- ✅ Deep learning-based image classification (`best_model.h5`)
- ✅ Emergency condition handling (e.g., low battery)
- ✅ Simulated drone behavior based on the detected class
- ✅ Battery management simulation (charging/discharging)
- ✅ Live camera feed using **DroidCam** via IP stream

---

## 📷 DroidCam Integration (Live Camera Feed)

We use [DroidCam](https://www.dev47apps.com/) to stream the phone camera to the Python script over Wi-Fi.

### 🔌 Setup Instructions:

1. **Install DroidCam** on your Android phone and connect it to the same Wi-Fi as your computer.
2. Open the app and note the **Wi-Fi IP** shown (e.g., `192.168.147.233`).
3. In the code (`main()` function), update the `VIDEO_SOURCE`:
   ```python
   VIDEO_SOURCE = "http://192.168.147.233:4747/video"
# Drone Navigation Web App
## 🚀 How to run:
```bash
pip install -r requirements.txt
python drone_nav_with_vision.py
