# YoloDetAndCls
Yolo11 detection and classification

项目结构：

--config: 用于存放模型训练的配置文件

--images: 存放测试用的图片

--ultralytics: Yolo项目库，由于ultralytics是个纯python项目，所以直接通过原文件引入更直观些

       
Yolo11部分的可以参考原项目：https://github.com/ultralytics/ultralytics


安装建议：

1、Python>=3.10  PyTorch>=1.8

2、如果使用GPU, 建议先基于官网安装好pytorch依赖： https://pytorch.org/

3、执行： pip install ultralytics

请参考 det_and_cls.py 中的使用示例