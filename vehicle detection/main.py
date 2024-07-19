import cv2
import numpy as np
import os

cfg_file = r"F:\python projects\vehicle detection\yolov3.cfg"
weights_file = r"F:\python projects\vehicle detection\yolov3.weights"
output_video_file = r"F:\python projects\vehicle detection\output_video.mp4"
output_dir = r"F:\python projects\vehicle detection\images"

# Load YOLO
print("Loading YOLO model...")
net = cv2.dnn.readNet(weights_file, cfg_file)
if net.empty():
    print("Failed to load YOLO model.")
    exit()
else:
    print("YOLO model loaded successfully.")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
classes = []
with open(r"F:\python projects\vehicle detection\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(f"Loaded {len(classes)} classes.")

# Load video
video_file = r'F:\python projects\vehicle detection\video1.mp4'
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video properties: FPS={fps}, Width={width}, Height={height}")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))
if video_writer.isOpened():
    print("VideoWriter is open and ready.")
else:
    print("Error: VideoWriter is not opened.")
    exit()

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames or failed to read frame.")
        break  # Exit the loop if no more frames

    print(f"Processing frame {frame_count}")

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        print(f"Detection output shape: {out.shape}")
        for detection in out:
            for obj in detection:
                if obj.ndim == 1:
                    obj = obj.reshape(-1)
                
                if obj.size >= 85:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Debug print for object values
                    print(f"Object detection: class_id={class_id}, confidence={confidence}, obj={obj[:10]}...")

                    if confidence > 0.2:  # Adjust confidence threshold as needed
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

    # Debug print for boxes and confidences
    print(f"Detected boxes: {boxes}")
    print(f"Detected confidences: {confidences}")
    print(f"Detected class_ids: {class_ids}")

    # Non-max suppression to remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.1, nms_threshold=0.4)  # Lowered threshold

    if len(indices) > 0:
        indices = indices.flatten()  # Flatten the array of indices
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Color for bounding box

            # Debug: Print details before drawing
            print(f"Drawing bounding box: {label} ({confidence:.2f}) at x={x}, y={y}, w={w}, h={h}")

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        print("No bounding boxes to draw.")

    # Save frame as image
    image_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    if cv2.imwrite(image_path, frame):
        print(f"Saved image: {image_path}")
    else:
        print(f"Failed to save image: {image_path}")

    frame_count += 1

    # Write frame to video
    if video_writer.isOpened():
        print(f"Writing frame {frame_count} to video")
        video_writer.write(frame)
    else:
        print("Error: video_writer is not a cv2.VideoWriter instance")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
