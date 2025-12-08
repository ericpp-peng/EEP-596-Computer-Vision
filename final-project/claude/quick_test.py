"""
快速測試版本 - 用現有模型先跑起來
不需要訓練新模型,直接用 COCO sports ball
"""

import torch
import os

# 設定環境變數以處理 PyTorch 2.6+ weights_only 問題
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import pickle

# === 配置 ===
VIDEO_PATH = "20250711_short.mp4"
OUTPUT_PATH = "quick_test_output.mp4"
CORNERS_FILE = "court_corners.pkl"

# === 載入模型 ===
print("載入模型...")

try:
    pose_model = YOLO("yolo11n-pose.pt")
    print("✅ 使用 YOLO11n-pose")
except:
    pose_model = YOLO("yolov8n-pose.pt")
    print("✅ 使用 YOLOv8n-pose")

det_model = YOLO("yolov8n.pt")

# === 載入球場角點 ===
COURT_CORNERS = None
if os.path.exists(CORNERS_FILE):
    with open(CORNERS_FILE, 'rb') as f:
        COURT_CORNERS = pickle.load(f)
    print(f"✅ 載入球場角點")

# === 工具函式 ===
def angle_3pt(a, b, c):
    """計算角度"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def draw_court(frame, corners):
    """畫球場"""
    if corners is None:
        return frame
    corners = np.array(corners, dtype=np.int32)
    cv2.polylines(frame, [corners], True, (255, 100, 0), 2)
    # 中線
    mid_top = ((corners[0] + corners[1]) // 2).tolist()
    mid_bot = ((corners[3] + corners[2]) // 2).tolist()
    cv2.line(frame, mid_top, mid_bot, (255, 100, 0), 2)
    return frame

# === 主程式 ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 找不到影片: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print(f"\n處理影片: {total} frames @ {fps} fps")
print("=" * 60)

frame_id = 0
ball_history = deque(maxlen=10)
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    
    # === Pose ===
    pose_res = pose_model(frame, imgsz=640)[0]
    
    keypoints = None
    if len(pose_res.keypoints) > 0:
        kpts = pose_res.keypoints.xy[0].cpu().numpy()
        keypoints = kpts
        
        # 畫骨架
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    
    # === 球偵測 (COCO sports ball = class 32) ===
    det_res = det_model(frame, imgsz=640, classes=[32, 38, 0])[0]
    
    ball_pos = None
    for box in det_res.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1+x2)/2, (y1+y2)/2
        
        if cls_id == 32:  # sports ball
            ball_pos = (cx, cy)
            cv2.circle(frame, (int(cx), int(cy)), 10, (0, 255, 255), -1)
            cv2.putText(frame, "Ball", (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # === 球場 ===
    frame = draw_court(frame, COURT_CORNERS)
    
    # === 擊球偵測 ===
    if ball_pos:
        ball_history.append(ball_pos)
    
    is_impact = False
    if keypoints is not None and len(ball_history) >= 2:
        # 右手腕
        r_wrist = keypoints[10]
        
        # 球與手腕距離
        if ball_pos:
            dist = np.linalg.norm(np.array(ball_pos) - r_wrist)
            ball_speed = np.linalg.norm(np.array(ball_history[-1]) - np.array(ball_history[-2]))
            
            if dist < 80 and ball_speed > 10:
                is_impact = True
    
    # === 動作分析 ===
    if is_impact and keypoints is not None:
        r_sh = keypoints[6]
        r_el = keypoints[8]
        r_wr = keypoints[10]
        r_hip = keypoints[12]
        
        elbow_ang = angle_3pt(r_sh, r_el, r_wr)
        shoulder_ang = angle_3pt(r_el, r_sh, r_hip)
        
        # 簡易分類
        action = "SMASH" if shoulder_ang > 70 and elbow_ang < 40 else "OTHER"
        
        # 顯示
        cv2.putText(frame, f"IMPACT! {action}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, f"Shoulder: {shoulder_ang:.1f} deg", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Elbow: {elbow_ang:.1f} deg", (50, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 評分
        score = 100
        if abs(shoulder_ang - 85) > 15:
            score -= 30
        if abs(elbow_ang - 160) > 20:
            score -= 25
        
        cv2.putText(frame, f"Score: {score}/100", (50, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # === 進度 ===
    cv2.putText(frame, f"Frame: {frame_id}/{total}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(frame)
    
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
