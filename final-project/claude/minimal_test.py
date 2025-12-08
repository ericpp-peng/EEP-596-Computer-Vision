"""
最小測試 - 羽球骨架偵測
"""

import torch
import torch.nn as nn

# 添加所有需要的類別
from ultralytics.nn.tasks import PoseModel, DetectionModel

safe_classes = [
    PoseModel,
    DetectionModel,
    nn.modules.container.Sequential,
    nn.modules.conv.Conv2d,
    nn.modules.batchnorm.BatchNorm2d,
    nn.modules.activation.SiLU,
    nn.modules.pooling.MaxPool2d,
    nn.modules.upsampling.Upsample,
    nn.modules.linear.Linear,
]

with torch.serialization.safe_globals(safe_classes):
    from ultralytics import YOLO
    print("載入模型...")
    pose_model = YOLO("yolov8n-pose.pt")
    print("✅ 模型載入成功!")

import cv2
import numpy as np

# 測試單幀
cap = cv2.VideoCapture("20250711_short.mp4")
ret, frame = cap.read()
cap.release()

if ret:
    print("執行推理...")
    results = pose_model(frame, imgsz=640)[0]
    
    if len(results.keypoints) > 0:
        kpts = results.keypoints.xy[0].cpu().numpy()
        print(f"✅ 偵測到骨架! {len(kpts)} 個關鍵點")
        
        # 畫骨架
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        cv2.imwrite("test_frame.jpg", frame)
        print("✅ 測試圖片已儲存: test_frame.jpg")
    else:
        print("⚠️  未偵測到人物")
else:
    print("❌ 無法讀取影片")
