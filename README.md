from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLOv8 ที่เทรนมาแล้ว (coco dataset)
model = YOLO("yolov8n.pt")  # n = nano (เบาที่สุด)

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับวัตถุ
    results = model(frame)

    # วาดผลลัพธ์บนภาพ
    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
