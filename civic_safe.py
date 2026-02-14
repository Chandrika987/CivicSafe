from ultralytics import YOLO
import cv2
from collections import deque

model = YOLO("yolov8n.pt")
# 🔥 Fire Detection Function
def detect_fire(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire color range (orange/yellow/red)
    lower = (18, 50, 50)
    upper = (35, 255, 255)

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Count bright fire pixels
    fire_pixels = cv2.countNonZero(mask)

    # Threshold for fire alert
    if fire_pixels > 4000:
        return True
    return False


def crowd_detection(threshold=5):
    cap = cv2.VideoCapture(0)

    # 🔹 Buffer to smooth count (prevents flickering)
    count_buffer = deque(maxlen=10)  # adjust 5–15 if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lower confidence so sitting / partial bodies are detected
        results = model(frame, stream=True, conf=0.4)

        raw_count = 0  # count for THIS frame only

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # ✔ Detect standing + sitting + partial people
                if cls == 0 and conf >= 0.4:
                    raw_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}",
                                (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

        # 🔹 Add to buffer & compute smooth count
        count_buffer.append(raw_count)
        person_count = round(sum(count_buffer) / len(count_buffer))
        fire_alert = detect_fire(frame)

        # 🔥 Check for fire
        if detect_fire(frame):
            cv2.putText(frame, "FIRE ALERT!", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


        # Alert logic
        color = (0, 255, 0)
        alert = "NORMAL"

        if person_count >= threshold:
            color = (0, 0, 255)
            alert = "CROWD ALERT!"

        cv2.putText(frame, f"People: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, alert, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("CivicSafe - Crowd Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

crowd_detection(threshold=2)
