#!/usr/bin/env python3
"""
羽球標註影像擷取工具
從影片中智能擷取最適合標註的 30 張圖片

使用方法：
    python extract_frames_for_annotation.py --video <影片路徑> --output badminton_ball_dataset/images

特點：
    - 自動選擇清晰、亮度合適的幀
    - 避免連續重複的幀
    - 平均分佈在整個影片中
    - 20 張 → train / 10 張 → val
"""

import cv2
import os
import numpy as np
import argparse
from pathlib import Path


def calculate_frame_quality(frame):
    """
    計算幀的質量分數（越高越好）
    考慮因素：清晰度、亮度、對比度
    """
    # 轉換為灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. 清晰度：使用 Laplacian 變異數
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. 亮度：平均值（避免過暗或過亮）
    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(brightness - 128) / 128  # 最佳亮度在 128 附近
    
    # 3. 對比度：標準差
    contrast = np.std(gray)
    
    # 綜合分數
    quality_score = (
        laplacian_var * 0.5 +      # 清晰度權重 50%
        brightness_score * 100 +    # 亮度權重
        contrast * 0.5              # 對比度權重
    )
    
    return quality_score


def extract_diverse_frames(video_path, num_frames=30, output_dir="badminton_ball_dataset/images"):
    """
    從影片中擷取多樣化的高質量幀
    
    Args:
        video_path: 影片檔案路徑
        num_frames: 要擷取的總幀數（預設 30）
        output_dir: 輸出目錄
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 無法開啟影片：{video_path}")
        return
    
    # 取得影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📹 影片資訊：")
    print(f"   總幀數：{total_frames}")
    print(f"   FPS：{fps}")
    print(f"   長度：{total_frames/fps:.2f} 秒")
    print()
    
    # 將影片分成多個區段，確保多樣性
    segment_size = total_frames // num_frames
    min_gap = max(fps // 2, 10)  # 至少間隔 0.5 秒或 10 幀
    
    selected_frames = []
    frame_qualities = []
    
    print(f"🔍 正在分析影片質量...")
    
    # 在每個區段中選擇最佳幀
    for i in range(num_frames):
        # 定義當前區段的範圍
        start_frame = i * segment_size
        end_frame = min((i + 1) * segment_size, total_frames)
        
        # 在區段內採樣
        sample_interval = max((end_frame - start_frame) // 10, 1)
        
        best_quality = -1
        best_frame_idx = start_frame
        best_frame_img = None
        
        for frame_idx in range(start_frame, end_frame, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # 計算質量
            quality = calculate_frame_quality(frame)
            
            if quality > best_quality:
                best_quality = quality
                best_frame_idx = frame_idx
                best_frame_img = frame.copy()
        
        if best_frame_img is not None:
            selected_frames.append((best_frame_idx, best_frame_img))
            frame_qualities.append(best_quality)
            print(f"   區段 {i+1}/{num_frames}: 幀 {best_frame_idx} (質量: {best_quality:.2f})")
    
    cap.release()
    
    # 分配到 train 和 val
    num_train = int(num_frames * 0.67)  # 67% 給 train (約 20 張)
    num_val = num_frames - num_train    # 33% 給 val (約 10 張)
    
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 儲存圖片...")
    print(f"   Train: {num_train} 張")
    print(f"   Val: {num_val} 張")
    print()
    
    # 儲存圖片
    for idx, (frame_idx, frame_img) in enumerate(selected_frames):
        if idx < num_train:
            save_dir = train_dir
            prefix = "train"
        else:
            save_dir = val_dir
            prefix = "val"
        
        filename = f"{prefix}_{idx:04d}_frame{frame_idx:06d}.jpg"
        save_path = save_dir / filename
        
        cv2.imwrite(str(save_path), frame_img)
        print(f"   ✓ {save_path}")
    
    print(f"\n✅ 完成！已擷取 {len(selected_frames)} 張圖片")
    print(f"\n📂 圖片位置：")
    print(f"   Train: {train_dir}")
    print(f"   Val: {val_dir}")
    print(f"\n🎯 下一步：使用 LabelImg 標註這些圖片")
    print(f"   指令：labelImg {train_dir}")
    print(f"   記得選擇 YOLO 格式！")


def main():
    parser = argparse.ArgumentParser(
        description="從羽球影片中擷取最適合標註的幀",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
    # 從預設影片擷取 30 張
    python extract_frames_for_annotation.py
    
    # 指定影片和數量
    python extract_frames_for_annotation.py --video pro/Chou-pro.mp4 --num 40
    
    # 指定輸出目錄
    python extract_frames_for_annotation.py --output my_dataset/images
        """
    )
    
    parser.add_argument(
        "--video", 
        type=str, 
        default="pro/Chou-pro.mp4",
        help="影片路徑（預設：pro/Chou-pro.mp4）"
    )
    
    parser.add_argument(
        "--num", 
        type=int, 
        default=30,
        help="要擷取的幀數（預設：30）"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="badminton_ball_dataset/images",
        help="輸出目錄（預設：badminton_ball_dataset/images）"
    )
    
    args = parser.parse_args()
    
    # 檢查影片是否存在
    if not os.path.exists(args.video):
        print(f"❌ 找不到影片：{args.video}")
        print(f"\n可用的影片：")
        # 搜尋常見位置
        for pattern in ["*.mp4", "*.avi", "*.mov", "pro/*.mp4"]:
            import glob
            videos = glob.glob(pattern)
            for v in videos:
                print(f"   - {v}")
        return
    
    print(f"🎬 開始處理影片：{args.video}")
    print(f"🎯 目標幀數：{args.num}")
    print(f"📁 輸出目錄：{args.output}")
    print("-" * 60)
    print()
    
    extract_diverse_frames(args.video, args.num, args.output)


if __name__ == "__main__":
    main()
