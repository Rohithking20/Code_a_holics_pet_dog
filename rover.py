import cv2

# Load YOLO model and configuration
net = cv2.dnn.readNet(r"C:\Users\rohit\OneDrive\Documents\Hackbattle\yolov3.weights", 
                      r"C:\Users\rohit\OneDrive\Documents\Hackbattle\yolov3.cfg")
layer_names = net.getLayerNames()

# Fix for IndexError in getUnconnectedOutLayers
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO dataset object names (80 classes)
with open(r"C:\Users\rohit\OneDrive\Documents\Hackbattle\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects_from_camera():
    # Initialize webcam
    cap = cv2.VideoCapture(1)  # 0 is the default webcam; change to 1 or 2 if using external cameras
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read from the webcam.")
            break

        height, width, channels = frame.shape

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Analyze detections
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]

                if confidence > 0.5:  # Only consider detections with confidence > 50%
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Bounding box coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-max suppression to remove duplicate boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Check if indices are not empty
        if indices is not None and len(indices) > 0:
            for i in indices.flatten():  # Use .flatten() to handle the scalar case
                box = boxes[i]
                x, y, w, h = box
                label = classes[class_ids[i]]
                confidence = confidences[i]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the frame with detections
        cv2.imshow("YOLO Object Detection", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start detection from the live camera
detect_objects_from_camera()
