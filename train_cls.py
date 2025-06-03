from ultralytics import YOLO
"""
    分类模型训练
"""
# Load a model
model = YOLO("config/hand_cls.yaml").load("model/yolo11x-cls.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="dataset/hand_cls", epochs=200, imgsz=224, name='hand_cls')