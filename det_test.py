from ultralytics import YOLO

# Load a model
model = YOLO('model/hand11x.pt').to('cuda:2')  # load a custom model

# Predict with the model
results = model("images/B1_langren_00001.jpg")  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each 
    result.save(filename="result.jpg")