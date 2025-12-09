#!/usr/bin/env python3
"""
從影片中手動選擇特定時間點擷取圖片
專門選擇羽球清楚、沒被遮擋的畫面
"""

import cv2
import os
from pathlib import Path

def extract_specific_frames(video_path, output_dir_train, output_dir_val):
    """擷取特定時間點的幀（專門選擇羽球清楚的時刻）"""
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 手動選擇的時間點（秒），專門選擇羽球清楚的時刻
    # 這些時間點會有多個連續幀，增加找到清楚羽球的機會
    selected_times = [
        2, 3, 5, 7, 9, 11, 13, 15, 17, 19,
        21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
        41, 43, 45, 47, 49, 51, 53, 55, 57, 59,
        61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85
    ]
    
    # 為每個時間點擷取 3 幀（前、中、後），增加找到清楚球的機會
    frames_to_extract = []
    for t in selected_times:
        frame_num = int(t * fps)
        frames_to_extract.extend([frame_num - 5, frame_num, frame_num + 5])
    
    # 去重並排序
    frames_to_extract = sorted(set(frames_to_extract))
    
    # 計算 train/val 分配
    num_train = int(len(frames_to_extract) * 0.67)
    
    print(f"🎬 準備擷取 {len(frames_to_extract)} 張圖片")
    print(f"   Train: {num_train} 張")
    print(f"   Val: {len(frames_to_extract) - num_train} 張")
    print()
    
    # 取得已有的最大編號
    existing_train = list(Path(output_dir_train).glob("train_*.jpg"))
    existing_val = list(Path(output_dir_val).glob("val_*.jpg"))
    
    train_start_idx = len(existing_train)
    val_start_idx = len(existing_val)
    
    train_count = 0
    val_count = 0
    
    for i, frame_num in enumerate(frames_to_extract):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # 決定存到 train 或 val
        if i < num_train:
            output_dir = output_dir_train
            idx = train_start_idx + train_count
            prefix = "train"
            train_count += 1
        else:
            output_dir = output_dir_val
            idx = val_start_idx + val_count
            prefix = "val"
            val_count += 1
        
        filename = f"{prefix}_{idx:04d}_frame{frame_num:06d}.jpg"
        save_path = Path(output_dir) / filename
        
        cv2.imwrite(str(save_path), frame)
        print(f"   ✓ {save_path}")
    
    cap.release()
    
    print(f"\n✅ 完成！新增：")
    print(f"   Train: {train_count} 張 (總共: {train_start_idx + train_count})")
    print(f"   Val: {val_count} 張 (總共: {val_start_idx + val_count})")
    print(f"\n🎯 請繼續標註新增的圖片")
    print(f"   Train: python simple_annotator.py train")
    print(f"   Val: python simple_annotator.py val")


if __name__ == "__main__":
    extract_specific_frames(
        "20250711_short.mp4",
        "badminton_ball_dataset/images/train",
        "badminton_ball_dataset/images/val"
    )
