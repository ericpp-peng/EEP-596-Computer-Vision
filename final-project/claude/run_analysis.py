"""
羽球骨架分析 - 完整版
直接 patch torch.load 來避免 weights_only 問題
"""

import torch
# Patch torch.load 以使用 weights_only=False
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO
import cv2
import numpy as np
import time

print("=" * 60)
print("羽球動作分析 - 快速測試版")
print("=" * 60)

# 載入模型
print("\n載入模型...")
pose_model = YOLO("yolov8n-pose.pt")
det_model = YOLO("yolov8n.pt")
print("✅ 模型載入完成\n")

# 開啟影片
VIDEO_PATH = "20250711_short.mp4"
OUTPUT_PATH = "quick_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 無法開啟影片: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print(f"處理影片: {total} frames @ {fps} fps ({w}x{h})")
print(f"輸出: {OUTPUT_PATH}\n")
print("=" * 60)

frame_id = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    
    # Pose 偵測
    pose_res = pose_model(frame, imgsz=640)[0]
    
    # 畫骨架
    if len(pose_res.keypoints) > 0:
        kpts = pose_res.keypoints.xy[0].cpu().numpy()
        
        for i, (x, y) in enumerate(kpts):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                # 標示關鍵關節
                if i in [6, 8, 10]:  # R_shoulder, R_elbow, R_wrist
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), 2)
    
    # 顯示資訊
    cv2.putText(frame, f"Frame: {frame_id}/{total}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    out.write(frame)
    
    # 每100幀顯示進度
    if frame_id % 100 == 0:
        elapsed = time.time() - start_time
        eta = (elapsed / frame_id) * (total - frame_id)
        fps_proc = frame_id / elapsed
        print(f"進度: {frame_id:5d}/{total} ({frame_id/total*100:5.1f}%) | "
              f"速度: {fps_proc:5.1f} fps | ETA: {int(eta//60):02d}:{int(eta%60):02d}")

cap.release()
out.release()

total_time = time.time() - start_time
print("\n" + "=" * 60)
print(f"✅ 完成!")
print(f"   輸出檔案: {OUTPUT_PATH}")
print(f"   處理時間: {int(total_time//60):02d}:{int(total_time%60):02d}")
print(f"   處理速度: {total/total_time:.1f} fps")
print("=" * 60)
print(f"\n執行以下指令查看結果:")
print(f"   open {OUTPUT_PATH}")
