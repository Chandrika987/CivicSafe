from ultralytics import YOLO
import cv2
from collections import deque
import winsound
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# 🔥 Fire Detection Function
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = (18, 50, 50)
    upper = (35, 255, 255)

    mask = cv2.inRange(hsv, lower, upper)
    fire_pixels = cv2.countNonZero(mask)

    return fire_pixels > 4000


# 🔔 Play alert sound in background
def play_alert_sound():
    winsound.PlaySound("alert.wav",
                       winsound.SND_FILENAME | winsound.SND_ASYNC)


def crowd_detection(threshold=5):
    cap = cv2.VideoCapture(0)

    count_buffer = deque(maxlen=10)

    last_alert_time = 0
    alert_cooldown = 5   # seconds (prevents continuous sound)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True, conf=0.4)

        raw_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf >= 0.4:
                    raw_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}",
                                (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

        # Smooth count
        count_buffer.append(raw_count)
        person_count = round(sum(count_buffer) / len(count_buffer))

        # Detect fire (only once!)
        fire_alert = detect_fire(frame)

        # Default status
        color = (0, 255, 0)
        alert_text = "NORMAL"
        alert_triggered = False

        # 🔥 Fire alert
        if fire_alert:
            cv2.putText(frame, "FIRE ALERT!", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
            alert_triggered = True

        # 👥 Crowd alert
        if person_count >= threshold:
            color = (0, 0, 255)
            alert_text = "CROWD ALERT!"
            alert_triggered = True

        # 🔔 Sound with cooldown
        current_time = time.time()
        if alert_triggered and (current_time - last_alert_time > alert_cooldown):
            print("ALERT TRIGGERED")
            play_alert_sound()
            last_alert_time = current_time

        # Display info
        cv2.putText(frame, f"People: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        cv2.putText(frame, alert_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        cv2.imshow("CivicSafe - Crowd Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()