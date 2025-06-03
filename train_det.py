from ultralytics import YOLO

# Load a model
model = YOLO("model/yolo11x.pt") # build from YAML and transfer weights

# Train the model, nothing to do with the results
results = model.train(data="config/body_detect.yaml",task='detect', epochs=100, imgsz=640, name='det_train_c2_x', batch=32)
