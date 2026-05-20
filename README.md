# 🚨 CivicSafe – AI-Based Crowd & Fire Detection System

CivicSafe is a real-time AI-powered surveillance and public safety monitoring system that detects overcrowding and fire-like incidents using computer vision techniques.

Built using **YOLOv8**, **OpenCV**, and **Python**, the system monitors live webcam feeds and triggers alerts during unsafe situations.

---

# 📌 Features

- 👥 Real-time crowd detection using YOLOv8
- 🔥 Fire detection using HSV color filtering
- 📹 Live webcam monitoring
- 🔔 Audio alert system with cooldown control
- 📊 Person count smoothing using deque buffer
- 🟢 Live bounding box visualization
- ⚡ Lightweight and fast inference using YOLOv8n

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Core programming language |
| OpenCV | Video capture & image processing |
| YOLOv8 (Ultralytics) | Real-time person detection |
| NumPy | Numerical computations |
| Winsound | Audio alert system |

---

# 📂 Project Structure

```bash
CivicSafe/
│
├── civic_safe.py        # Core detection logic
├── app_civic.py         # Main execution file
├── test_sound.py        # Alert sound testing
├── alert.wav            # Alert audio file
├── yolov8n.pt           # YOLOv8 model
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/CivicSafe.git
cd CivicSafe
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python ultralytics numpy
```

---

# ▶️ How to Run

## Run the Main Project

```bash
python app_civic.py
```

---

## Test Alert Sound (Optional)

```bash
python test_sound.py
```

---

# 🧠 System Workflow

```text
Webcam Input
     ↓
YOLOv8 Person Detection
     ↓
Person Counting + Smoothing
     ↓
Crowd Threshold Check
     ↓
Fire Detection (HSV Color Filtering)
     ↓
Alert Decision Logic
     ↓
Audio Alert System
     ↓
Live Display Window
```

---

# 👥 Crowd Detection

- Uses YOLOv8n pretrained model
- Detects persons in real-time
- Counts people from webcam feed
- Smooths count using deque buffer
- Triggers crowd alert if threshold exceeds limit

---

# 🔥 Fire Detection

The system uses HSV color filtering to identify fire-like regions.

### Process:
1. Convert frame to HSV color space
2. Apply fire color range mask
3. Count fire-like pixels
4. Trigger fire alert if threshold exceeds limit

---

# 🔔 Alert System

- Alert sound plays asynchronously
- Cooldown mechanism prevents repeated alarms
- Alerts triggered for:
  - Crowd threshold exceeded
  - Fire detection

---

# 📌 Advantages

- Real-time monitoring
- Lightweight AI model
- Fast detection speed
- Low-cost implementation
- Easy deployment

---

# ⚠️ Limitations

- Fire detection is color-based
- Windows-only sound support (`winsound`)
- No cloud/database integration
- Single camera support
- No remote notification system

---

# 🔮 Future Enhancements

- Deep learning-based fire detection
- SMS/Email alert integration
- Cloud-based incident logging
- Multi-camera support
- Web dashboard integration
- Mobile application support

---

# 📋 Requirements

Contents of `requirements.txt`:

```txt
opencv-python
ultralytics
numpy
```

---

# 💻 System Requirements

## Hardware
- Webcam
- Minimum 4GB RAM
- Intel i5 or higher recommended

## Software
- Python 3.9+
- Windows OS (recommended)

---

# 📖 Conclusion

CivicSafe demonstrates how AI and computer vision can improve public safety monitoring through automated crowd detection and fire alert systems.

The project serves as a strong prototype for smart surveillance and future smart-city safety applications.

---

