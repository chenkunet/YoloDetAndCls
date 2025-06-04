import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def draw_boxes(image, boxes_info, box_color=(0,255,0), font_scale=0.7, thickness=2):
    """
    在图片上绘制检测/分类结果的box和类别
    
    boxes_info: 
        [{'box': [x1, y1, x2, y2], 'cls_name':类别名, 'conf':置信度}, ...]
    返回：绘制后的图片
    """
    img = image.copy()
    for obj in boxes_info:
        x1, y1, x2, y2 = obj['box']
        cls_name = obj.get('cls_name', 'object')
        conf = obj.get('conf', 0)
        label = f'{cls_name} {conf:.2f}'
        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        # 画label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1-th-4), (x1+tw, y1), box_color, -1)
        cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
    return img

class CascadeYoloDetAndCls:
    def __init__(self, detect_weight, classify_weight, det_target=['hand'], device='cuda:0'):
        """
        detect_weight: 检测模型权重文件
        classify_weight: 分类模型权重文件
        det_target: 需要进一步分类的detect类别名
        device: 可选, 指定cuda或cpu
        """
        self.det_model = YOLO(str(detect_weight)).to(device)
        self.cls_model = YOLO(str(classify_weight)).to(device)
        self.det_target = det_target

    def predict_image(self, image, min_box=5):
        """
        image: 路径or ndarray
        min_box: 检测框最小边 shorter than 该值将跳过
        返回：每个目标的{'det_box', 'cls_name', 'cls_conf', 'cls_id'} 以及检测类别
        """
        # 加载图片
        img0 = cv2.imread(image) if isinstance(image, (str, Path)) else image.copy()
        h0, w0 = img0.shape[:2]
        results = self.det_model(img0)
        det_result = results[0]
        res = []
        for i in range(len(det_result.boxes.cls)):
            name = det_result.names[int(det_result.boxes.cls[i])]
            xyxy = det_result.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(x2, w0-1), min(y2, h0-1)
            conf = float(det_result.boxes.conf[i])
            # 跳过太小目标
            if min(x2-x1, y2-y1) < min_box:
                continue

            if name not in self.det_target:
                res.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'cls_name': name,
                    'conf': conf
                })
                continue

            crop = img0[y1:y2, x1:x2]
            # 分类推理
            if crop.shape[0]<2 or crop.shape[1]<2:
                continue
            cls_res = self.cls_model(crop)[0]
            # 分类最高置信度
            probs = cls_res.probs.cpu().numpy().data
            topid = int(np.argmax(probs))
            conf = float(probs[topid])
            res.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'cls_name': cls_res.names[topid] if hasattr(cls_res,"names") else str(topid),
                'conf': conf
            })
        return res
    
    def infer_video(self, video_path, save_path, frame_interval=5, min_box=5, box_color=(0,255,0)):
        """
        对视频进行推理，每frame_interval帧推理一次，其他帧复用上次结果。
        输入:
            video_path: 原视频文件路径
            save_path: 保存推理结果视频文件路径
            frame_interval: 每隔多少帧推理一次
        """
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

        last_result = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 对间隔帧推理
            if frame_id % frame_interval == 0:
                last_result = self.predict_image(frame, min_box=min_box)

            # 绘制结果
            vis_img = draw_boxes(frame, last_result, box_color=box_color)
            out.write(vis_img)

            frame_id += 1

        cap.release()
        out.release()
        print(f"推理完成，已保存：{save_path}")


if __name__ == "__main__":
    # 单图分析
    images = cv2.imread('images/B1_langren_00001.jpg')
    model = CascadeYoloDetAndCls(detect_weight='model/hand11s.pt', classify_weight='model/cls.pt', device='cuda:0')
    res = model.predict_image(images)
    print(res)
    images = draw_boxes(images, res)
    # 视频分析
    cv2.imwrite("result.jpg", images)
    model.infer_video(video_path='images/test_video.mp4',save_path='result_video.mp4')