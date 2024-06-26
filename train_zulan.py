from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
# model = YOLO("yolov8n.pt") 
model = YOLO("runs/detect/train9/weights/last.pt") 

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="datasets/FindZulan-9/data.yaml", epochs=200, imgsz=640)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format="onnx")