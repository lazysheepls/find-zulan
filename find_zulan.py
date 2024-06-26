import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
# model = YOLO("yolov8n.pt")
model = YOLO("/home/yang/Documents/Playground/play_yolo/find_zulan/runs/detect/train10/weights/best.pt")

# Define remote image or video URL
# source = "https://www.bilibili.com/video/BV1HK4y1N718/?spm_id_from=333.999.0.0&vd_source=d531a4ab9358864c713e21e0248418f2"
# source = "https://youtu.be/LNwODJXcvt4"

# Video capture
cap = cv2.VideoCapture("source.mp4")

# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to read camera feed")

# Set resolutions in int
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4))
size = (frame_width, frame_height) 

# Video writer
is_write = False

if is_write:
    output = cv2.VideoWriter('annotated_vid.mp4',  
                         cv2.VideoWriter_fourcc(*'MP4V'), 
                         30, size) 

while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()

    if success:
        # Run inference on the source
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write frame
        if is_write:
            output.write(annotated_frame) 

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
if is_write:
    output.release()

cv2.destroyAllWindows()
