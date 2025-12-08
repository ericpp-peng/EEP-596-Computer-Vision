"""
從影片中提取幀用於標注
自動選擇羽球可能出現的幀
"""

import cv2
import os
import numpy as np
from pathlib import Path

# === 配置 ===
VIDEO_PATH = "20250711_short.mp4"
OUTPUT_DIR = "annotation_images"
NUM_FRAMES = 100  # 要提取的幀數

# 建立輸出目錄
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("=" * 60)
print("準備標注圖片")
print("=" * 60)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"影片: {VIDEO_PATH}")
print(f"總幀數: {total_frames}")
print(f"FPS: {fps:.2f}")
print(f"目標提取: {NUM_FRAMES} 幀")
print("=" * 60)

# 均勻取樣
frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

saved_count = 0
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    # 儲存
    filename = f"frame_{idx:06d}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, frame)
    
    saved_count += 1
    if saved_count % 10 == 0:
        print(f"已提取: {saved_count}/{NUM_FRAMES}")

cap.release()

print("=" * 60)
print(f"✅ 完成! 已提取 {saved_count} 張圖片到: {OUTPUT_DIR}/")
print("=" * 60)
print("\n接下來:")
print("1. 使用標注工具標記羽球:")
print("   - Roboflow (推薦，線上): https://roboflow.com")
print("   - LabelImg (本地): pip install labelImg")
print("   - CVAT (進階): https://cvat.ai")
print()
print("2. 標注類別:")
print("   - shuttlecock (羽球)")
print()
print("3. 匯出為 YOLOv8 格式")
print("4. 執行 train_shuttlecock.py 開始訓練")
print("=" * 60)
