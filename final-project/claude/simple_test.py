"""
快速測試版 - 羽球動作分析
"""

import torch
from ultralytics.nn.tasks import PoseModel, DetectionModel

with torch.serialization.safe_globals([PoseModel, DetectionModel]):
    from ultralytics import YOLO

import cv2
import numpy as np
import time
from collections import deque

print("載入模型...")
pose_model = YOLO("yolov8n-pose.pt")
det_model = YOLO("yolov8n.pt")
print("✅ 模型載入完成")

# 開啟影片
VIDEO_PATH = "20250711_short.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ 無法開啟影片: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

OUTPUT_PATH = "quick_test_output.mp4"
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print(f"\n處理影片: {total} frames @ {fps} fps")
print("=" * 60)

frame_id = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    
    # Pose 偵測
    pose_res = pose_model(frame, imgsz=640)[0]
    
    if len(pose_res.keypoints) > 0:
        kpts = pose_res.keypoints.xy[0].cpu().numpy()
        
        # 畫骨架點
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # 顯示 frame 資訊
    cv2.putText(frame, f"Frame: {frame_id}/{total}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    out.write(frame)
    
    # 進度顯示
    if frame_id % 100 == 0:
        elapsed = time.time() - start
        eta = (elapsed / frame_id) * (total - frame_id)
        print(f"進度: {frame_id}/{total} ({frame_id/total*100:.1f}%) | ETA: {eta:.1f}s")

cap.release()
out.release()

total_time = time.time() - start
print("\n" + "=" * 60)
print(f"✅ 完成!")
print(f"輸出: {OUTPUT_PATH}")
print(f"處理時間: {total_time:.1f} 秒")
print(f"速度: {total/total_time:.1f} fps")
print("=" * 60)
