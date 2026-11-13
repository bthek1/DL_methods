# Car detection

Car Detection

Author

Benedict Thekkel

``` python
from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8s.pt' with 'yolov8n.pt' for a lighter model if necessary

# Dictionary to filter and label specific objects
target_classes = model.names

# Reference the camera directly by device path
cap = cv2.VideoCapture('/dev/video0')  # Replace '/dev/video0' with your device path if different

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(frame, conf=0.5, imgsz=128)

    # Loop through results and only show detections for specified classes
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            class_id = int(box.cls)
            if class_id in target_classes:
                label_name = target_classes[class_id]  # Get the label from the dictionary
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates
                confidence = box.conf[0]  # Confidence score
                label = f"{label_name} {confidence:.2f}"

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Targeted Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

    0: 96x128 1 person, 8.5ms
    Speed: 0.3ms preprocess, 8.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.6ms
    Speed: 0.5ms preprocess, 7.6ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.2ms
    Speed: 0.4ms preprocess, 7.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 6.7ms
    Speed: 0.4ms preprocess, 6.7ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.4ms
    Speed: 0.5ms preprocess, 9.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 6.7ms
    Speed: 0.4ms preprocess, 6.7ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 1.4ms preprocess, 10.6ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.2ms
    Speed: 0.5ms preprocess, 11.2ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.1ms
    Speed: 0.5ms preprocess, 11.1ms inference, 1.0ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.1ms
    Speed: 0.5ms preprocess, 11.1ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.1ms
    Speed: 0.4ms preprocess, 11.1ms inference, 0.9ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.6ms
    Speed: 0.4ms preprocess, 8.6ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.4ms
    Speed: 0.6ms preprocess, 8.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.5ms
    Speed: 0.4ms preprocess, 7.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.4ms
    Speed: 0.6ms preprocess, 7.4ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.4ms
    Speed: 0.6ms preprocess, 7.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.1ms
    Speed: 0.5ms preprocess, 8.1ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.8ms
    Speed: 0.6ms preprocess, 7.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.8ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.6ms preprocess, 10.5ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.6ms preprocess, 10.6ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.7ms
    Speed: 0.5ms preprocess, 8.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.8ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.5ms preprocess, 10.2ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.9ms preprocess, 10.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.8ms preprocess, 10.5ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.5ms preprocess, 10.8ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.1ms
    Speed: 0.6ms preprocess, 12.1ms inference, 0.9ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.8ms
    Speed: 0.5ms preprocess, 8.8ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.4ms
    Speed: 0.5ms preprocess, 9.4ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.5ms
    Speed: 0.6ms preprocess, 8.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.9ms
    Speed: 0.6ms preprocess, 7.9ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.0ms
    Speed: 0.4ms preprocess, 9.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.7ms
    Speed: 0.6ms preprocess, 9.7ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.7ms
    Speed: 0.5ms preprocess, 11.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.5ms preprocess, 10.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.5ms preprocess, 11.0ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.5ms preprocess, 7.7ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.6ms
    Speed: 0.6ms preprocess, 11.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.5ms preprocess, 10.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.6ms
    Speed: 0.5ms preprocess, 8.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.4ms
    Speed: 0.4ms preprocess, 7.4ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.6ms preprocess, 7.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.6ms preprocess, 10.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.8ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.2ms
    Speed: 0.6ms preprocess, 11.2ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.5ms preprocess, 10.5ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.9ms
    Speed: 0.5ms preprocess, 12.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.9ms preprocess, 10.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.2ms
    Speed: 0.8ms preprocess, 12.2ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.6ms preprocess, 10.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.7ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.6ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.8ms preprocess, 10.4ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.7ms preprocess, 11.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.9ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.1ms
    Speed: 0.5ms preprocess, 9.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.5ms preprocess, 10.2ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.2ms
    Speed: 0.6ms preprocess, 11.2ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.1ms
    Speed: 0.4ms preprocess, 7.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 6.9ms
    Speed: 0.5ms preprocess, 6.9ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.7ms preprocess, 7.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.2ms
    Speed: 0.7ms preprocess, 7.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.2ms
    Speed: 0.4ms preprocess, 7.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.8ms
    Speed: 0.7ms preprocess, 7.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.3ms
    Speed: 0.8ms preprocess, 12.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.6ms
    Speed: 0.7ms preprocess, 12.6ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.1ms
    Speed: 0.6ms preprocess, 12.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.6ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.7ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.7ms preprocess, 10.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.6ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.6ms
    Speed: 0.6ms preprocess, 9.6ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 6.6ms
    Speed: 0.4ms preprocess, 6.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.6ms
    Speed: 0.6ms preprocess, 9.6ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.6ms preprocess, 10.8ms inference, 1.0ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.6ms
    Speed: 0.6ms preprocess, 7.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.5ms
    Speed: 0.4ms preprocess, 7.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.9ms
    Speed: 0.8ms preprocess, 7.9ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.7ms
    Speed: 0.6ms preprocess, 9.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.7ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.7ms preprocess, 10.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.6ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.4ms preprocess, 7.7ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.4ms
    Speed: 0.7ms preprocess, 8.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.3ms
    Speed: 0.5ms preprocess, 7.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.2ms
    Speed: 0.6ms preprocess, 7.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.8ms preprocess, 10.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.6ms preprocess, 10.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.6ms
    Speed: 0.4ms preprocess, 7.6ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.9ms
    Speed: 0.6ms preprocess, 9.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.9ms
    Speed: 0.5ms preprocess, 9.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.1ms
    Speed: 0.7ms preprocess, 7.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 6.8ms
    Speed: 0.6ms preprocess, 6.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.1ms
    Speed: 0.5ms preprocess, 7.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.0ms
    Speed: 0.7ms preprocess, 7.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.7ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.2ms
    Speed: 0.8ms preprocess, 8.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.0ms
    Speed: 0.4ms preprocess, 7.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.7ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.1ms
    Speed: 0.7ms preprocess, 8.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.4ms
    Speed: 0.5ms preprocess, 7.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.8ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.3ms
    Speed: 0.8ms preprocess, 11.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.7ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.6ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.5ms preprocess, 7.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.5ms
    Speed: 0.6ms preprocess, 7.5ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.9ms
    Speed: 0.4ms preprocess, 9.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.8ms preprocess, 10.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.7ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.8ms
    Speed: 0.6ms preprocess, 8.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.2ms
    Speed: 0.6ms preprocess, 8.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.6ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.9ms
    Speed: 0.6ms preprocess, 8.9ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.6ms
    Speed: 0.6ms preprocess, 7.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.6ms preprocess, 7.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.9ms
    Speed: 0.6ms preprocess, 7.9ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.6ms preprocess, 10.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.4ms
    Speed: 0.6ms preprocess, 9.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.6ms
    Speed: 0.6ms preprocess, 7.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.5ms
    Speed: 0.7ms preprocess, 7.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.1ms
    Speed: 0.6ms preprocess, 8.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.6ms preprocess, 7.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.5ms preprocess, 7.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.5ms
    Speed: 0.7ms preprocess, 8.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.6ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.9ms
    Speed: 0.6ms preprocess, 11.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.6ms
    Speed: 0.6ms preprocess, 8.6ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.4ms preprocess, 10.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.5ms preprocess, 10.8ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.6ms preprocess, 11.0ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.2ms
    Speed: 0.6ms preprocess, 11.2ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.7ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.2ms
    Speed: 0.8ms preprocess, 11.2ms inference, 1.0ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.8ms
    Speed: 0.5ms preprocess, 7.8ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.6ms
    Speed: 0.5ms preprocess, 11.6ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.3ms
    Speed: 0.7ms preprocess, 7.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.5ms
    Speed: 0.5ms preprocess, 8.5ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.5ms
    Speed: 0.5ms preprocess, 9.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.3ms
    Speed: 0.6ms preprocess, 9.3ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.5ms
    Speed: 0.5ms preprocess, 7.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.6ms preprocess, 7.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.9ms
    Speed: 0.7ms preprocess, 7.9ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.5ms
    Speed: 0.7ms preprocess, 8.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.6ms preprocess, 7.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.8ms
    Speed: 0.5ms preprocess, 8.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.5ms
    Speed: 0.5ms preprocess, 8.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.5ms
    Speed: 0.6ms preprocess, 8.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.7ms
    Speed: 0.8ms preprocess, 8.7ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.3ms
    Speed: 0.6ms preprocess, 9.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.8ms
    Speed: 0.5ms preprocess, 7.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.8ms
    Speed: 0.8ms preprocess, 7.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.6ms
    Speed: 0.6ms preprocess, 8.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.2ms
    Speed: 0.5ms preprocess, 8.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.8ms
    Speed: 0.5ms preprocess, 7.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.3ms
    Speed: 0.8ms preprocess, 8.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.9ms
    Speed: 0.6ms preprocess, 9.9ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.8ms
    Speed: 0.6ms preprocess, 8.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.5ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.5ms preprocess, 10.8ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.5ms preprocess, 10.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.6ms preprocess, 10.8ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.6ms preprocess, 10.2ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.6ms preprocess, 10.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.0ms
    Speed: 0.5ms preprocess, 9.0ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.1ms
    Speed: 0.5ms preprocess, 11.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.6ms
    Speed: 0.5ms preprocess, 8.6ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.4ms
    Speed: 0.6ms preprocess, 8.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.7ms
    Speed: 0.7ms preprocess, 8.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.4ms
    Speed: 0.5ms preprocess, 8.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.5ms
    Speed: 0.3ms preprocess, 8.5ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.8ms
    Speed: 0.5ms preprocess, 9.8ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.7ms
    Speed: 0.4ms preprocess, 7.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.6ms
    Speed: 0.6ms preprocess, 8.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.5ms preprocess, 10.9ms inference, 0.9ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.7ms preprocess, 11.0ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.8ms preprocess, 10.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 6.9ms
    Speed: 0.5ms preprocess, 6.9ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.7ms
    Speed: 0.6ms preprocess, 11.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.7ms preprocess, 10.4ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.7ms preprocess, 10.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.7ms preprocess, 10.8ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.3ms
    Speed: 0.3ms preprocess, 7.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 13.5ms
    Speed: 0.7ms preprocess, 13.5ms inference, 0.9ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.3ms
    Speed: 0.7ms preprocess, 11.3ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.1ms
    Speed: 0.4ms preprocess, 11.1ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.6ms
    Speed: 0.8ms preprocess, 8.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.4ms
    Speed: 0.6ms preprocess, 7.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.7ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.6ms
    Speed: 0.6ms preprocess, 11.6ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.6ms preprocess, 10.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 13.4ms
    Speed: 0.8ms preprocess, 13.4ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 9.9ms
    Speed: 0.9ms preprocess, 9.9ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.4ms
    Speed: 0.5ms preprocess, 10.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.4ms
    Speed: 0.8ms preprocess, 12.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.5ms
    Speed: 0.6ms preprocess, 12.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 7.9ms
    Speed: 0.4ms preprocess, 7.9ms inference, 0.3ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 7.1ms
    Speed: 0.6ms preprocess, 7.1ms inference, 0.2ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.6ms
    Speed: 0.9ms preprocess, 10.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.0ms
    Speed: 0.6ms preprocess, 10.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 13.1ms
    Speed: 0.9ms preprocess, 13.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 13.1ms
    Speed: 0.5ms preprocess, 13.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 7.2ms
    Speed: 0.4ms preprocess, 7.2ms inference, 0.2ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.6ms
    Speed: 0.8ms preprocess, 10.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.5ms
    Speed: 0.8ms preprocess, 10.5ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.5ms
    Speed: 0.9ms preprocess, 12.5ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.6ms
    Speed: 0.7ms preprocess, 11.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 14.9ms
    Speed: 0.8ms preprocess, 14.9ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 16.0ms
    Speed: 0.8ms preprocess, 16.0ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 15.2ms
    Speed: 0.7ms preprocess, 15.2ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.3ms
    Speed: 0.8ms preprocess, 11.3ms inference, 0.3ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.9ms
    Speed: 0.8ms preprocess, 10.9ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.6ms
    Speed: 0.9ms preprocess, 10.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.4ms
    Speed: 1.0ms preprocess, 10.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.6ms
    Speed: 0.8ms preprocess, 10.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.5ms
    Speed: 0.9ms preprocess, 11.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.0ms
    Speed: 0.6ms preprocess, 12.0ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 7.5ms
    Speed: 0.4ms preprocess, 7.5ms inference, 0.3ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.3ms
    Speed: 0.4ms preprocess, 10.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 9.5ms
    Speed: 0.7ms preprocess, 9.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.4ms
    Speed: 0.7ms preprocess, 12.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.0ms
    Speed: 1.0ms preprocess, 11.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 13.1ms
    Speed: 0.7ms preprocess, 13.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.7ms preprocess, 10.7ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.7ms preprocess, 11.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.7ms preprocess, 10.4ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.9ms
    Speed: 0.6ms preprocess, 11.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.5ms preprocess, 10.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 9.3ms
    Speed: 0.6ms preprocess, 9.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 13.1ms
    Speed: 0.9ms preprocess, 13.1ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.9ms preprocess, 10.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.8ms preprocess, 11.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.5ms
    Speed: 0.6ms preprocess, 7.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.9ms preprocess, 10.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.7ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.7ms preprocess, 10.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.7ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.8ms preprocess, 10.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.7ms preprocess, 10.8ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.7ms preprocess, 11.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.8ms
    Speed: 0.9ms preprocess, 11.8ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.5ms
    Speed: 0.8ms preprocess, 11.5ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.6ms
    Speed: 0.7ms preprocess, 11.6ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.1ms
    Speed: 0.7ms preprocess, 11.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.9ms
    Speed: 0.6ms preprocess, 11.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.8ms
    Speed: 0.8ms preprocess, 11.8ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.7ms
    Speed: 0.8ms preprocess, 11.7ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.3ms
    Speed: 0.5ms preprocess, 8.3ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.8ms
    Speed: 0.7ms preprocess, 9.8ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.9ms preprocess, 10.4ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.7ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.8ms preprocess, 10.2ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 14.3ms
    Speed: 0.6ms preprocess, 14.3ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.3ms
    Speed: 0.4ms preprocess, 7.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.8ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.7ms preprocess, 10.5ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.7ms preprocess, 10.6ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.5ms
    Speed: 0.9ms preprocess, 8.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.8ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.9ms
    Speed: 0.7ms preprocess, 9.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.3ms
    Speed: 0.5ms preprocess, 8.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.6ms preprocess, 10.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.7ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.7ms preprocess, 10.8ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.8ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.7ms preprocess, 10.5ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.8ms preprocess, 10.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.8ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.1ms
    Speed: 0.8ms preprocess, 11.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.5ms
    Speed: 0.5ms preprocess, 11.5ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.5ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.8ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.8ms preprocess, 10.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.8ms preprocess, 10.5ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.3ms
    Speed: 0.7ms preprocess, 12.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.4ms
    Speed: 0.6ms preprocess, 7.4ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.9ms
    Speed: 0.7ms preprocess, 9.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.8ms preprocess, 10.4ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.9ms
    Speed: 0.8ms preprocess, 11.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.8ms preprocess, 10.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.9ms preprocess, 11.0ms inference, 1.0ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.7ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.9ms
    Speed: 0.9ms preprocess, 11.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.3ms
    Speed: 0.6ms preprocess, 11.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.8ms preprocess, 10.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.7ms
    Speed: 0.7ms preprocess, 9.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.4ms
    Speed: 0.5ms preprocess, 9.4ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.5ms preprocess, 10.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.9ms preprocess, 10.2ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.4ms
    Speed: 0.5ms preprocess, 7.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.8ms preprocess, 10.9ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.4ms
    Speed: 0.8ms preprocess, 11.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.6ms preprocess, 10.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.2ms
    Speed: 1.0ms preprocess, 9.2ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.6ms
    Speed: 0.8ms preprocess, 12.6ms inference, 0.9ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.6ms preprocess, 10.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.1ms
    Speed: 0.5ms preprocess, 9.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.8ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.9ms
    Speed: 0.6ms preprocess, 9.9ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.1ms
    Speed: 0.8ms preprocess, 9.1ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.5ms
    Speed: 0.9ms preprocess, 12.5ms inference, 1.0ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.3ms
    Speed: 0.6ms preprocess, 12.3ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.5ms
    Speed: 0.5ms preprocess, 7.5ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.8ms preprocess, 10.6ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.5ms
    Speed: 0.6ms preprocess, 10.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.7ms
    Speed: 0.9ms preprocess, 10.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.9ms
    Speed: 0.8ms preprocess, 10.9ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.6ms
    Speed: 0.8ms preprocess, 10.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.8ms
    Speed: 0.6ms preprocess, 12.8ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.9ms
    Speed: 0.8ms preprocess, 10.9ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.7ms
    Speed: 0.5ms preprocess, 11.7ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.5ms
    Speed: 0.6ms preprocess, 10.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.3ms
    Speed: 0.7ms preprocess, 10.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.7ms
    Speed: 0.7ms preprocess, 11.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.9ms
    Speed: 0.6ms preprocess, 10.9ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.8ms
    Speed: 0.7ms preprocess, 10.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.3ms
    Speed: 0.8ms preprocess, 10.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 7.4ms
    Speed: 0.4ms preprocess, 7.4ms inference, 0.2ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 8.6ms
    Speed: 0.4ms preprocess, 8.6ms inference, 0.2ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.4ms
    Speed: 0.6ms preprocess, 10.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 8.7ms
    Speed: 0.6ms preprocess, 8.7ms inference, 0.3ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 9.1ms
    Speed: 0.6ms preprocess, 9.1ms inference, 0.3ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.0ms
    Speed: 0.8ms preprocess, 12.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.8ms
    Speed: 0.6ms preprocess, 11.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.1ms
    Speed: 0.9ms preprocess, 11.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.2ms
    Speed: 0.5ms preprocess, 11.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.8ms
    Speed: 0.7ms preprocess, 10.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.8ms preprocess, 10.5ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.0ms
    Speed: 0.7ms preprocess, 10.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.4ms
    Speed: 0.7ms preprocess, 10.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.1ms
    Speed: 0.7ms preprocess, 12.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.1ms
    Speed: 0.8ms preprocess, 12.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 7.3ms
    Speed: 0.5ms preprocess, 7.3ms inference, 0.2ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.8ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.7ms preprocess, 10.7ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.7ms preprocess, 10.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.9ms preprocess, 10.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.7ms
    Speed: 0.7ms preprocess, 9.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 13.2ms
    Speed: 0.8ms preprocess, 13.2ms inference, 0.9ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.1ms
    Speed: 0.8ms preprocess, 10.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 8.9ms
    Speed: 0.7ms preprocess, 8.9ms inference, 0.2ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 1.0ms preprocess, 10.6ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.8ms preprocess, 10.8ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.7ms
    Speed: 0.8ms preprocess, 11.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.0ms
    Speed: 0.8ms preprocess, 12.0ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.6ms
    Speed: 0.5ms preprocess, 11.6ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.6ms
    Speed: 0.7ms preprocess, 11.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.4ms
    Speed: 0.4ms preprocess, 7.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.2ms
    Speed: 0.8ms preprocess, 10.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.3ms
    Speed: 0.7ms preprocess, 10.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.4ms
    Speed: 0.9ms preprocess, 10.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.1ms
    Speed: 0.8ms preprocess, 11.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.7ms
    Speed: 0.7ms preprocess, 10.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.3ms
    Speed: 0.9ms preprocess, 12.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.2ms
    Speed: 0.7ms preprocess, 11.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.1ms
    Speed: 0.9ms preprocess, 11.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.5ms
    Speed: 0.8ms preprocess, 11.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.0ms
    Speed: 0.7ms preprocess, 12.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.6ms
    Speed: 0.9ms preprocess, 11.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.6ms
    Speed: 1.0ms preprocess, 11.6ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.9ms
    Speed: 0.6ms preprocess, 10.9ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.3ms
    Speed: 0.9ms preprocess, 11.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 6.9ms
    Speed: 0.6ms preprocess, 6.9ms inference, 0.2ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.1ms
    Speed: 0.8ms preprocess, 11.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.3ms
    Speed: 0.8ms preprocess, 11.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.4ms
    Speed: 0.7ms preprocess, 12.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.0ms
    Speed: 0.5ms preprocess, 11.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.0ms
    Speed: 0.8ms preprocess, 11.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.5ms
    Speed: 0.7ms preprocess, 11.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 9.7ms
    Speed: 0.6ms preprocess, 9.7ms inference, 0.3ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 9.2ms
    Speed: 0.6ms preprocess, 9.2ms inference, 0.3ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 7.8ms
    Speed: 0.5ms preprocess, 7.8ms inference, 0.3ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.6ms preprocess, 10.9ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.9ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.7ms preprocess, 11.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.9ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.6ms preprocess, 11.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.1ms
    Speed: 0.6ms preprocess, 7.1ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.9ms preprocess, 10.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.7ms preprocess, 10.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.5ms preprocess, 10.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.5ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.9ms
    Speed: 0.6ms preprocess, 11.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.3ms
    Speed: 1.0ms preprocess, 12.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.1ms
    Speed: 0.6ms preprocess, 12.1ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.9ms
    Speed: 0.6ms preprocess, 11.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.2ms
    Speed: 0.7ms preprocess, 11.2ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.3ms
    Speed: 0.5ms preprocess, 11.3ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.9ms preprocess, 10.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.8ms preprocess, 10.6ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.6ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.8ms
    Speed: 0.7ms preprocess, 9.8ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.6ms
    Speed: 0.4ms preprocess, 7.6ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.3ms
    Speed: 0.5ms preprocess, 9.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.3ms
    Speed: 0.8ms preprocess, 12.3ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.8ms
    Speed: 0.3ms preprocess, 7.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.2ms
    Speed: 0.7ms preprocess, 7.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.9ms
    Speed: 0.5ms preprocess, 7.9ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.4ms
    Speed: 0.7ms preprocess, 11.4ms inference, 0.9ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.8ms preprocess, 10.9ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.6ms preprocess, 10.7ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 15.4ms
    Speed: 0.6ms preprocess, 15.4ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.4ms
    Speed: 0.5ms preprocess, 11.4ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.8ms
    Speed: 0.5ms preprocess, 7.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.6ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.4ms
    Speed: 0.3ms preprocess, 7.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.2ms
    Speed: 0.7ms preprocess, 11.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.2ms
    Speed: 0.4ms preprocess, 7.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.8ms preprocess, 10.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.8ms preprocess, 10.6ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.6ms preprocess, 10.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.5ms preprocess, 10.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.7ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.7ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.6ms
    Speed: 0.6ms preprocess, 10.6ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.8ms
    Speed: 0.5ms preprocess, 9.8ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.9ms
    Speed: 0.7ms preprocess, 9.9ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.8ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.9ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.9ms
    Speed: 0.8ms preprocess, 11.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.8ms
    Speed: 0.5ms preprocess, 7.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.2ms
    Speed: 0.8ms preprocess, 10.2ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.9ms preprocess, 10.4ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.9ms preprocess, 10.9ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.7ms preprocess, 10.4ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.2ms
    Speed: 0.8ms preprocess, 10.2ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 8.0ms
    Speed: 0.5ms preprocess, 8.0ms inference, 0.2ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.0ms
    Speed: 0.7ms preprocess, 11.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.3ms
    Speed: 0.8ms preprocess, 11.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 11.5ms
    Speed: 0.7ms preprocess, 11.5ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 12.2ms
    Speed: 0.9ms preprocess, 12.2ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 (no detections), 10.7ms
    Speed: 0.6ms preprocess, 10.7ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.4ms preprocess, 10.4ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.7ms preprocess, 10.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.7ms preprocess, 10.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 2 persons, 8.0ms
    Speed: 0.7ms preprocess, 8.0ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.0ms
    Speed: 0.8ms preprocess, 10.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.8ms preprocess, 10.7ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.6ms preprocess, 11.0ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.5ms
    Speed: 0.9ms preprocess, 10.5ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.8ms preprocess, 10.8ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.5ms
    Speed: 0.4ms preprocess, 9.5ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.3ms
    Speed: 0.6ms preprocess, 10.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.7ms
    Speed: 0.4ms preprocess, 9.7ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.4ms
    Speed: 0.5ms preprocess, 8.4ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.8ms preprocess, 11.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.3ms
    Speed: 0.8ms preprocess, 11.3ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.3ms
    Speed: 0.6ms preprocess, 9.3ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.7ms
    Speed: 0.6ms preprocess, 9.7ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.8ms
    Speed: 0.8ms preprocess, 10.8ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.0ms
    Speed: 0.6ms preprocess, 8.0ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.4ms
    Speed: 0.6ms preprocess, 8.4ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.3ms
    Speed: 0.6ms preprocess, 7.3ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.7ms
    Speed: 0.5ms preprocess, 8.7ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.7ms preprocess, 10.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 9.3ms
    Speed: 0.7ms preprocess, 9.3ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.4ms
    Speed: 0.6ms preprocess, 10.4ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.1ms
    Speed: 0.9ms preprocess, 11.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.7ms preprocess, 10.9ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.1ms
    Speed: 0.7ms preprocess, 12.1ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.7ms
    Speed: 0.5ms preprocess, 10.7ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 12.0ms
    Speed: 0.7ms preprocess, 12.0ms inference, 0.8ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.0ms
    Speed: 0.6ms preprocess, 11.0ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.8ms
    Speed: 0.3ms preprocess, 8.8ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.8ms
    Speed: 0.6ms preprocess, 8.8ms inference, 0.4ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 7.5ms
    Speed: 0.4ms preprocess, 7.5ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 8.4ms
    Speed: 0.5ms preprocess, 8.4ms inference, 0.5ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.1ms
    Speed: 0.7ms preprocess, 10.1ms inference, 0.6ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 10.9ms
    Speed: 0.8ms preprocess, 10.9ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

    0: 96x128 1 person, 11.7ms
    Speed: 0.7ms preprocess, 11.7ms inference, 0.7ms postprocess per image at shape (1, 3, 96, 128)

``` python
from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8s.pt' with 'yolov8n.pt' for a lighter model if necessary

# Print the full dictionary of class labels
print("Class ID to Label Mapping:")
for class_id, label in model.names.items():
    print(f"{class_id}: {label}")
```

    Class ID to Label Mapping:
    0: person
    1: bicycle
    2: car
    3: motorcycle
    4: airplane
    5: bus
    6: train
    7: truck
    8: boat
    9: traffic light
    10: fire hydrant
    11: stop sign
    12: parking meter
    13: bench
    14: bird
    15: cat
    16: dog
    17: horse
    18: sheep
    19: cow
    20: elephant
    21: bear
    22: zebra
    23: giraffe
    24: backpack
    25: umbrella
    26: handbag
    27: tie
    28: suitcase
    29: frisbee
    30: skis
    31: snowboard
    32: sports ball
    33: kite
    34: baseball bat
    35: baseball glove
    36: skateboard
    37: surfboard
    38: tennis racket
    39: bottle
    40: wine glass
    41: cup
    42: fork
    43: knife
    44: spoon
    45: bowl
    46: banana
    47: apple
    48: sandwich
    49: orange
    50: broccoli
    51: carrot
    52: hot dog
    53: pizza
    54: donut
    55: cake
    56: chair
    57: couch
    58: potted plant
    59: bed
    60: dining table
    61: toilet
    62: tv
    63: laptop
    64: mouse
    65: remote
    66: keyboard
    67: cell phone
    68: microwave
    69: oven
    70: toaster
    71: sink
    72: refrigerator
    73: book
    74: clock
    75: vase
    76: scissors
    77: teddy bear
    78: hair drier
    79: toothbrush

``` python
```

    {0: 'person',
     1: 'bicycle',
     2: 'car',
     3: 'motorcycle',
     4: 'airplane',
     5: 'bus',
     6: 'train',
     7: 'truck',
     8: 'boat',
     9: 'traffic light',
     10: 'fire hydrant',
     11: 'stop sign',
     12: 'parking meter',
     13: 'bench',
     14: 'bird',
     15: 'cat',
     16: 'dog',
     17: 'horse',
     18: 'sheep',
     19: 'cow',
     20: 'elephant',
     21: 'bear',
     22: 'zebra',
     23: 'giraffe',
     24: 'backpack',
     25: 'umbrella',
     26: 'handbag',
     27: 'tie',
     28: 'suitcase',
     29: 'frisbee',
     30: 'skis',
     31: 'snowboard',
     32: 'sports ball',
     33: 'kite',
     34: 'baseball bat',
     35: 'baseball glove',
     36: 'skateboard',
     37: 'surfboard',
     38: 'tennis racket',
     39: 'bottle',
     40: 'wine glass',
     41: 'cup',
     42: 'fork',
     43: 'knife',
     44: 'spoon',
     45: 'bowl',
     46: 'banana',
     47: 'apple',
     48: 'sandwich',
     49: 'orange',
     50: 'broccoli',
     51: 'carrot',
     52: 'hot dog',
     53: 'pizza',
     54: 'donut',
     55: 'cake',
     56: 'chair',
     57: 'couch',
     58: 'potted plant',
     59: 'bed',
     60: 'dining table',
     61: 'toilet',
     62: 'tv',
     63: 'laptop',
     64: 'mouse',
     65: 'remote',
     66: 'keyboard',
     67: 'cell phone',
     68: 'microwave',
     69: 'oven',
     70: 'toaster',
     71: 'sink',
     72: 'refrigerator',
     73: 'book',
     74: 'clock',
     75: 'vase',
     76: 'scissors',
     77: 'teddy bear',
     78: 'hair drier',
     79: 'toothbrush'}

``` python
a = [1,2,3]

b = [2,3,4]

c = None

result = min([i for i in [a,b,c] if i])
print(result)
```

    [1, 2, 3]

Back to top
