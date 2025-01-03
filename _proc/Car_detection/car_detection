from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8s.pt' with 'yolov8n.pt' for a lighter model if necessary

# Dictionary to filter and label specific objects
target_classes = {
    0: 'person',  # COCO ID for 'person'
    2: 'car',     # COCO ID for 'car'
    1: 'bicycle'  # COCO ID for 'bicycle'
}

# Reference the camera directly by device path
cap = cv2.VideoCapture('/dev/video0')  # Replace '/dev/video0' with your device path if different

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(frame, conf=0.5, imgsz=320)

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
