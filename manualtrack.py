import cv2
import numpy as np

# Load YOLOv3 network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names from COCO dataset used by YOLOv3
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

selected_box = None
boxes = []
confidences = []
class_ids = []

def detect_and_display(frame):
    global boxes, confidences, class_ids
    height, width = frame.shape[:2]

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes.clear()
    confidences.clear()
    class_ids.clear()

    # Loop over each detection
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == "person":  # Filter for persons
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])

        # Draw the bounding box for the detected person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detection boxes
    cv2.imshow('Frame', frame)


def click_event(event, x_click, y_click, flags, param):
    global boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is inside any of the detected boxes
        for i, (x, y, w, h) in enumerate(boxes):
            if x <= x_click <= x + w and y <= y_click <= y + h:
                print(f"Person clicked at box: {x, y, w, h}")
                # Save screenshot of the clicked person
                frame = param  # This will now contain the current frame
                screenshot = frame[y:y + h, x:x + w]
                cv2.imwrite(f'screenshot_yolo_person_{i}.png', screenshot)
                print(f'Screenshot saved as screenshot_yolo_person_{i}.png')
                break  # Exit after detecting one person to prevent multiple screenshots


# Capture video from file or camera
cap = cv2.VideoCapture('3318088-hd_1920_1080_25fps.mp4')  # Replace 'video.mp4' with 0 for webcam

cv2.namedWindow('Frame')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects and display the frame
    detect_and_display(frame)

    # Set the callback to handle mouse clicks, pass the current frame
    cv2.setMouseCallback('Frame', click_event, frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
