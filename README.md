# Python Real-Time Object Detection

A Python application that uses a pretrained YOLO model to detect and count objects in real time using a webcam.

## Features

- Real-time object detection through a webcam
- Bounding boxes around detected objects
- Object labels and confidence scores
- Live counting of detected objects by category

## Technologies Used

- **Python** — Main programming language
- **OpenCV** — Captures and displays webcam footage
- **Ultralytics** — Provides access to the YOLO object detection model
- **YOLO26 Nano** — Pretrained neural network for object detection
- **Counter** — Python utility used to count detected objects

## Installation

Install the required Python libraries:

```bash
pip install opencv-python ultralytics
```

## Usage

Run the application:

```bash
python camera.py
```

The webcam will open and display detected objects with bounding boxes, labels, confidence scores and object counts.

Press **Escape** to close the application.

The pretrained YOLO model will be downloaded automatically if it is not already available.

## What I Learned

This project helped me develop my understanding of computer vision and how pretrained machine learning models can be integrated into Python applications.

I also explored how factors such as lighting can affect object detection performance.
