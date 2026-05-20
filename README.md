# CivicSafe 🛡️

A real-time AI-powered public safety system that detects crowd overcrowding and fire hazards simultaneously using a live webcam feed — and triggers instant audio alerts.

---

## 🔍 What It Does

CivicSafe monitors live video and raises alerts for two types of safety risks:

| Detection | Method | Alert Trigger |
|---|---|---|
| 👥 Crowd | YOLOv8 person detection | When crowd count ≥ threshold |
| 🔥 Fire | OpenCV HSV color masking | When fire-colored pixels > 4000 |

Both run in real time on a single webcam feed with a shared audio alert system (with 5-second cooldown to avoid repeated sounds).

---

## 🧠 How It Works

```
Live Webcam Feed
      │
      ├──► YOLOv8 (Person Detection)
      │         └── Rolling 10-frame average → Crowd Alert
      │
      └──► OpenCV HSV Masking
                └── Pixel threshold check → Fire Alert
                          │
                    🔔 Audio Alert (alert.wav) with cooldown
```

- **Smooth counting**: a 10-frame rolling buffer prevents flickering crowd counts from noisy detections
- **Confidence filtering**: only detections with ≥ 40% confidence are counted
- **Modular code**: `civic_safe.py` holds all logic; `app_civic.py` is the clean entry point

---

## 🛠️ Tech Stack

- Python 3.x
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) — real-time object detection
- OpenCV — video capture and HSV fire detection
- NumPy — array operations

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Chandrika987/CivicSafe.git
cd CivicSafe
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python app_civic.py
```

Press `q` to quit the live window.

---

## ⚙️ Configuration

In `app_civic.py`, adjust the crowd threshold:

```python
crowd_detection(threshold=5)  # Alert when 5+ people detected
```

---

## 📁 Project Structure

```
CivicSafe/
├── civic_safe.py       # Core detection logic (crowd + fire + alerts)
├── app_civic.py        # Entry point
├── alert.wav           # Alert sound file
├── yolov8n.pt          # Pre-trained YOLOv8 nano model
├── requirements.txt
└── README.md
```

---

## 🔮 Planned Improvements (Ongoing)

- [ ] Streamlit web dashboard for live monitoring
- [ ] Hugging Face model integration for smarter scene classification
- [ ] Multi-camera support
- [ ] Logging alerts to a database with timestamps
- [ ] Deploy as a portable desktop app

---

## 👩‍💻 Author

**Chandrika Pala** — B.Tech CSE, SRKR Engineering College  
[LinkedIn](https://linkedin.com/in/chandrikapala) · [GitHub](https://github.com/Chandrika987)
