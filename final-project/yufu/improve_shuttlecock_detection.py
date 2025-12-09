#!/usr/bin/env python3
"""
改進羽球偵測 - 完整工作流程腳本

這個腳本會幫你：
1. 從目前的影片中擷取更多未被偵測到的羽球畫面
2. 自動找出「應該有球但沒偵測到」的幀
3. 整合到現有的資料集
4. 重新訓練模型

使用方法：
    # Step 1: 擷取新的標註圖片（從未偵測到球的幀）
    python improve_shuttlecock_detection.py --mode extract --video ./20250711_short.mp4
    
    # Step 2: 使用 simple_annotator.py 標註新圖片
    # 需要先合併到 train/val 再標註
    
    # Step 3: 重新訓練模型（包含新資料）
    python improve_shuttlecock_detection.py --mode train --epochs 100
"""

import cv2
import os
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import shutil
from datetime import datetime


def extract_missed_ball_frames(
    video_path="./20250711_short.mp4",
    existing_model="./runs/detect/shuttlecock_train/weights/best.pt",
    output_dir="badminton_ball_dataset/images/additional",
    num_frames=50,
    conf_threshold=0.25,
    include_detected=True  # 新增：是否包含已偵測到的幀
):
    """
    擷取適合標註的羽球畫面
    策略：
    1. 優先選擇未偵測到但有運動的幀（可能漏掉的球）
    2. 如果不夠，也包含已偵測到但品質好的幀（增加多樣性）
    """
    print("🔍 Step 1: 尋找適合標註的羽球畫面...")
    print(f"   影片：{video_path}")
    print(f"   模型：{existing_model}")
    print(f"   目標：擷取 {num_frames} 張圖片")
    print(f"   策略：{'包含已偵測到的幀' if include_detected else '只找未偵測的幀'}")
    print("-" * 60)
    
    # 載入現有模型
    model = YOLO(existing_model)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 無法開啟影片")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📹 影片資訊：{total_frames} 幀, {fps} FPS")
    
    # 建立輸出目錄
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 候選幀清單
    # 分成兩類：未偵測的（優先）和已偵測的（補充）
    missed_frames = []  # 未偵測但有運動的幀
    detected_frames = []  # 已偵測且品質好的幀
    prev_frame = None
    
    print("\n🎬 分析影片中...")
    for frame_idx in range(0, total_frames, 2):  # 每2幀檢查一次，加速處理
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 進度顯示
        if frame_idx % 100 == 0:
            print(f"   進度: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
        
        # 使用模型偵測
        results = model(frame, conf=conf_threshold, verbose=False)
        has_detection = len(results[0].boxes) > 0
        
        # 計算幀品質
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 如果沒有偵測到球，但有明顯運動（可能有球）
        if not has_detection and prev_frame is not None:
            # 計算幀差異（運動程度）
            diff = cv2.absdiff(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            )
            motion_score = np.mean(diff)
            
            # 如果有足夠運動且清晰度夠高，加入未偵測候選
            if motion_score > 5 and sharpness > 100:
                quality_score = motion_score * 2 + sharpness / 10  # 未偵測的給更高分
                missed_frames.append((frame_idx, quality_score, frame.copy()))
        
        # 如果有偵測到球，也加入候選（品質好的）
        elif has_detection and include_detected:
            if sharpness > 150:  # 只選清晰的
                quality_score = sharpness / 10
                detected_frames.append((frame_idx, quality_score, frame.copy()))
        
        prev_frame = frame
    
    cap.release()
    
    print(f"\n✅ 找到 {len(missed_frames)} 個未偵測幀")
    print(f"✅ 找到 {len(detected_frames)} 個已偵測幀")
    
    # 合併並按質量排序
    # 優先使用未偵測的，不夠的話補充已偵測的
    missed_frames.sort(key=lambda x: x[1], reverse=True)
    detected_frames.sort(key=lambda x: x[1], reverse=True)
    
    # 選擇策略：70% 未偵測 + 30% 已偵測（如果可以的話）
    num_missed = min(len(missed_frames), int(num_frames * 0.7))
    num_detected = min(len(detected_frames), num_frames - num_missed)
    
    selected_frames = missed_frames[:num_missed] + detected_frames[:num_detected]
    
    # 如果還不夠，全部用已偵測的補足
    if len(selected_frames) < num_frames and len(detected_frames) > num_detected:
        remaining = num_frames - len(selected_frames)
        selected_frames.extend(detected_frames[num_detected:num_detected + remaining])
    
    print(f"\n📸 選擇：")
    print(f"   - 未偵測幀：{min(len(missed_frames), num_missed)} 張")
    print(f"   - 已偵測幀：{len(selected_frames) - min(len(missed_frames), num_missed)} 張")
    print(f"   - 總計：{len(selected_frames)} 張")
    
    # 儲存選中的幀
    print(f"\n💾 儲存圖片到 {output_dir}...")
    for i, (frame_idx, score, frame) in enumerate(selected_frames):
        filename = f"missed_frame_{frame_idx:06d}_score{score:.0f}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        if (i + 1) % 10 == 0:
            print(f"   已儲存 {i+1}/{len(selected_frames)} 張")
    
    print(f"\n✨ 完成！請標註這些圖片：")
    print(f"\n下一步：")
    print(f"   1. 分配圖片：python improve_shuttlecock_detection.py --mode split")
    print(f"   2. 標註圖片：python simple_annotator.py train")
    print(f"   3. 訓練模型：python improve_shuttlecock_detection.py --mode train")


def split_images_for_annotation(additional_dir="badminton_ball_dataset/images/additional"):
    """
    將 additional 資料夾的圖片分配到 train/val（標註前準備）
    """
    print("\n📦 分配圖片到 train/val 資料夾...")
    
    additional_path = Path(additional_dir)
    if not additional_path.exists():
        print(f"❌ 找不到額外圖片目錄：{additional_dir}")
        return False
    
    # 取得所有圖片
    image_files = list(additional_path.glob("*.jpg")) + list(additional_path.glob("*.png"))
    if not image_files:
        print(f"❌ 在 {additional_dir} 中沒有找到圖片")
        return False
    
    print(f"   找到 {len(image_files)} 張圖片")
    
    # 目標資料夾
    train_img_dir = Path("badminton_ball_dataset/images/train")
    val_img_dir = Path("badminton_ball_dataset/images/val")
    
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    
    # 分配圖片
    train_count = 0
    val_count = 0
    
    for img_file in image_files:
        # 80% 去 train, 20% 去 val
        if np.random.random() < 0.8:
            target_dir = train_img_dir
            train_count += 1
        else:
            target_dir = val_img_dir
            val_count += 1
        
        # 複製檔案
        shutil.copy(img_file, target_dir / img_file.name)
    
    print(f"✅ 圖片分配完成：")
    print(f"   Train: {train_count} 張")
    print(f"   Val: {val_count} 張")
    print(f"\n下一步：標註圖片")
    print(f"   指令：python simple_annotator.py train")
    return True


def merge_and_prepare_dataset(additional_dir="badminton_ball_dataset/images/additional"):
    """
    合併新標註的圖片到現有資料集
    """
    print("\n📦 Step 2: 整合新標註資料...")
    
    additional_path = Path(additional_dir)
    if not additional_path.exists():
        print(f"❌ 找不到額外圖片目錄：{additional_dir}")
        return False
    
    # 檢查是否有標註檔
    label_files = list(additional_path.glob("*.txt"))
    if not label_files:
        print(f"⚠️  警告：在 {additional_dir} 中沒有找到標註檔 (.txt)")
        print(f"   請先使用 simple_annotator.py 標註圖片")
        return False
    
    print(f"   找到 {len(label_files)} 個標註檔")
    
    # 移動圖片和標註到 train 資料集（80% 機率）或 val（20%）
    train_img_dir = Path("badminton_ball_dataset/images/train")
    train_lbl_dir = Path("badminton_ball_dataset/labels/train")
    val_img_dir = Path("badminton_ball_dataset/images/val")
    val_lbl_dir = Path("badminton_ball_dataset/labels/val")
    
    moved_count = 0
    for label_file in label_files:
        img_file = label_file.with_suffix('.jpg')
        if not img_file.exists():
            img_file = label_file.with_suffix('.png')
        
        if img_file.exists():
            # 80% 去 train, 20% 去 val
            if np.random.random() < 0.8:
                target_img_dir = train_img_dir
                target_lbl_dir = train_lbl_dir
                dataset_type = "train"
            else:
                target_img_dir = val_img_dir
                target_lbl_dir = val_lbl_dir
                dataset_type = "val"
            
            # 複製檔案（保留原始檔案以防萬一）
            shutil.copy(img_file, target_img_dir / img_file.name)
            shutil.copy(label_file, target_lbl_dir / label_file.name)
            moved_count += 1
            
            if moved_count % 10 == 0:
                print(f"   已整合 {moved_count}/{len(label_files)} 個樣本")
    
    print(f"✅ 成功整合 {moved_count} 個新樣本到資料集")
    return True


def train_improved_model(epochs=100, batch_size=16, img_size=640):
    """
    使用擴充後的資料集重新訓練模型
    """
    print("\n🎯 Step 3: 重新訓練模型...")
    print(f"   訓練週期：{epochs}")
    print(f"   批次大小：{batch_size}")
    print(f"   圖片大小：{img_size}")
    print("-" * 60)
    
    # 檢查資料集
    data_yaml = "badminton_ball_dataset/data.yaml"
    if not Path(data_yaml).exists():
        print(f"❌ 找不到資料集設定檔：{data_yaml}")
        return
    
    # 載入預訓練模型
    model = YOLO("yolov8n.pt")
    
    # 訓練
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device='mps',  # 使用 M4 Pro GPU
        project='runs/detect',
        name=f'shuttlecock_improved_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        exist_ok=True,
        
        # 針對小物體優化
        patience=30,
        save=True,
        save_period=20,
        
        # 加強資料增強
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.2,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        
        # 針對小物體的特殊設定
        close_mosaic=10,  # 最後10個epoch關閉mosaic
    )
    
    print("\n✅ 訓練完成！")
    print(f"   最佳模型：runs/detect/shuttlecock_improved_*/weights/best.pt")
    print(f"\n💡 下一步：")
    print(f"   1. 複製最佳模型到工作目錄")
    print(f"   2. 更新 testPlayerPoseEst.py 中的 SHUTTLECOCK_WEIGHTS 路徑")
    print(f"   3. 重新執行影片分析")


def main():
    parser = argparse.ArgumentParser(description="改進羽球偵測工作流程")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['extract', 'split', 'merge', 'train', 'full'],
        default='extract',
        help="執行模式：extract=擷取圖片, split=分配圖片, merge=整合標註, train=訓練, full=完整流程"
    )
    parser.add_argument("--video", type=str, default="./20250711_short.mp4", help="影片路徑")
    parser.add_argument("--model", type=str, default="./runs/detect/shuttlecock_train/weights/best.pt", help="現有模型路徑")
    parser.add_argument("--num-frames", type=int, default=50, help="要擷取的圖片數量")
    parser.add_argument("--epochs", type=int, default=100, help="訓練週期")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏸 羽球偵測改進工具")
    print("=" * 60)
    
    if args.mode == 'extract':
        extract_missed_ball_frames(
            video_path=args.video,
            existing_model=args.model,
            num_frames=args.num_frames
        )
    
    elif args.mode == 'split':
        split_images_for_annotation()
    
    elif args.mode == 'merge':
        merge_and_prepare_dataset()
    
    elif args.mode == 'train':
        # 如果 additional 有標註就合併，沒有就直接訓練
        additional_path = Path("badminton_ball_dataset/images/additional")
        has_labels = len(list(additional_path.glob("*.txt"))) > 0 if additional_path.exists() else False
        
        if has_labels:
            if merge_and_prepare_dataset():
                train_improved_model(epochs=args.epochs, batch_size=args.batch)
        else:
            # 直接訓練（資料已經在 train/val 了）
            print("\n⏩ 跳過合併步驟，直接訓練")
            train_improved_model(epochs=args.epochs, batch_size=args.batch)
    
    elif args.mode == 'full':
        # 完整流程
        extract_missed_ball_frames(
            video_path=args.video,
            existing_model=args.model,
            num_frames=args.num_frames
        )
        print("\n" + "=" * 60)
        print("⏸️  請依序執行：")
        print("   1. 分配圖片：python improve_shuttlecock_detection.py --mode split")
        print("   2. 標註圖片：python simple_annotator.py train")
        print("   3. 訓練模型：python improve_shuttlecock_detection.py --mode train")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print("✨ 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
