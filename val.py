# -*- coding: utf-8 -*-

import utils.font_sync
import os
import torch
from ultralytics import YOLO
from utils.report import report_model_performance, print_table
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    # 模型路径
    yolo11s_pt = 'weights/yolo11s/weights/best.pt'
    yolo11s_SE_pt = 'weights/yolo11s-attention-SE/weights/best.pt'

    # 评估一个模型
    # model_infos = yolo11s_pt
    # 同时评估多个模型
    model_infos = [yolo11s_pt, yolo11s_SE_pt]
    # 保存所有模型的评估结果
    all_results = []
    # 数据集路径的yaml文件
    data_path = os.path.join('config', 'traindata.yaml')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not isinstance(model_infos, list):
        model_infos = [model_infos]
    for model_path in model_infos:
        print('\n现在评估的模式为：', model_path)
        model = YOLO(str(model_path))
        # 模型评估
        metrics = model.val(data=data_path,  # 数据集路径
                            imgsz=640,  # 图片大小，要和训练时一样
                            batch=2,  # batch
                            workers=0,  # 加载数据线程数
                            conf=0.001,  # 设置检测的最小置信度阈值。置信度低于此阈值的检测将被丢弃。
                            iou=0.6,  # 设置非最大抑制 (NMS) 的交叉重叠 (IoU) 阈值。有助于减少重复检测。
                            device=device,  # 使用显卡
                            project='runs/val',  # 保存路径
                            name='exp',  # 保存命名
                            split='val',  # 取值val/test
                            )
        report_model_performance(all_results, model_path, metrics, model)
    # 打印模型评估报告
    print_table(all_results)
