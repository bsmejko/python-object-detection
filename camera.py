import cv2
from ultralytics import YOLO
from collections import Counter

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            raise RuntimeError("Could not read frame from camera.")

        # Run object detection on the current frame
        results = model(frame, verbose=False)

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Count how many of each object were detected
        counts = Counter(
            model.names[int(class_id)]
            for class_id in results[0].boxes.cls.tolist()
        )

        # Display object counts in the top-left corner
        for index, (name, count) in enumerate(counts.items()):
            text = f"{name}: {count}"

            cv2.putText(
                annotated_frame,
                text,
                (10, 30 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # Display the webcam feed
        cv2.imshow("YOLO Object Detection", annotated_frame)

        # Press ESC to exit
        if cv2.waitKey(1) == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()