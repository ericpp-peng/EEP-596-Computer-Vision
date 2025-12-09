#!/usr/bin/env python3
"""
羽球偵測器訓練腳本
使用 YOLOv8 訓練自定義羽球偵測模型

使用方法：
    python train_shuttlecock_detector.py
    
    # 自訂訓練參數
    python train_shuttlecock_detector.py --epochs 100 --batch 16 --img 640
    
訓練後會產生：
    runs/detect/train/weights/best.pt  ← 最佳模型
    runs/detect/train/weights/last.pt  ← 最後模型
"""

import argparse
from ultralytics import YOLO
import torch
from pathlib import Path


def train_shuttlecock_detector(
    data_yaml="badminton_ball_dataset/data.yaml",
    model_name="yolov8n.pt",
    epochs=50,
    batch_size=16,
    img_size=640,
    device=None
):
    """
    訓練羽球偵測器
    
    Args:
        data_yaml: 資料集設定檔路徑
        model_name: 預訓練模型名稱
        epochs: 訓練週期數
        batch_size: 批次大小
        img_size: 輸入圖片大小
        device: 運算裝置 (None=自動, 'cpu', 'mps', '0'=GPU)
    """
    
    # 檢查資料集是否存在
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ 找不到資料集設定檔：{data_yaml}")
        print(f"\n請確認：")
        print(f"1. 已經用 extract_frames_for_annotation.py 擷取圖片")
        print(f"2. 已經用 simple_annotator.py 標註完成")
        print(f"3. 標註檔 (.txt) 已放在 labels/train 和 labels/val")
        return None
    
    # 自動偵測裝置
    if device is None:
        if torch.cuda.is_available():
            device = '0'
            print("🚀 使用 GPU (CUDA)")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("🚀 使用 Apple Silicon GPU (MPS)")
        else:
            device = 'cpu'
            print("💻 使用 CPU")
    
    print(f"\n📋 訓練設定：")
    print(f"   資料集：{data_yaml}")
    print(f"   基礎模型：{model_name}")
    print(f"   訓練週期：{epochs}")
    print(f"   批次大小：{batch_size}")
    print(f"   圖片大小：{img_size}")
    print(f"   運算裝置：{device}")
    print("-" * 60)
    
    # 載入模型
    print(f"\n📦 載入預訓練模型...")
    model = YOLO(model_name)
    
    # 開始訓練
    print(f"\n🎯 開始訓練...")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project='runs/detect',
        name='shuttlecock_train',
        exist_ok=True,
        
        # 優化小物體偵測
        patience=20,          # Early stopping patience
        save=True,            # 儲存檢查點
        save_period=10,       # 每 10 epoch 儲存一次
        
        # 資料增強（對小物體重要）
        hsv_h=0.015,          # 色調增強
        hsv_s=0.7,            # 飽和度增強
        hsv_v=0.4,            # 亮度增強
        degrees=10,           # 旋轉角度
        translate=0.1,        # 平移
        scale=0.5,            # 縮放
        shear=0.0,            # 剪切
        perspective=0.0,      # 透視
        flipud=0.0,           # 上下翻轉
        fliplr=0.5,           # 左右翻轉
        mosaic=1.0,           # Mosaic 增強
        mixup=0.0,            # Mixup 增強
        
        # 小物體優化
        cls=0.5,              # 分類損失權重
        box=7.5,              # 邊界框損失權重
        dfl=1.5,              # DFL 損失權重
    )
    
    # 訓練完成
    print(f"\n✅ 訓練完成！")
    print(f"\n📊 訓練結果：")
    print(f"   最佳模型：runs/detect/shuttlecock_train/weights/best.pt")
    print(f"   最後模型：runs/detect/shuttlecock_train/weights/last.pt")
    print(f"   訓練圖表：runs/detect/shuttlecock_train/")
    
    # 驗證模型
    print(f"\n🔍 驗證模型...")
    metrics = model.val()
    
    print(f"\n📈 驗證指標：")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    return model


def test_detector(model_path, test_video="pro/Chou-pro.mp4"):
    """
    測試訓練好的偵測器
    
    Args:
        model_path: 訓練好的模型路徑
        test_video: 測試影片路徑
    """
    print(f"\n🎬 測試偵測器...")
    print(f"   模型：{model_path}")
    print(f"   影片：{test_video}")
    
    model = YOLO(model_path)
    
    # 在影片上測試
    results = model.predict(
        source=test_video,
        save=True,
        conf=0.25,
        iou=0.45,
        show=False,
        project='runs/detect',
        name='shuttlecock_test',
        exist_ok=True
    )
    
    print(f"\n✅ 測試完成！")
    print(f"   結果影片：runs/detect/shuttlecock_test/")


def main():
    parser = argparse.ArgumentParser(
        description="訓練羽球偵測器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
    # 基本訓練（50 epochs）
    python train_shuttlecock_detector.py
    
    # 快速測試（10 epochs）
    python train_shuttlecock_detector.py --epochs 10
    
    # 高精度訓練（100 epochs，大批次）
    python train_shuttlecock_detector.py --epochs 100 --batch 32
    
    # 訓練後測試
    python train_shuttlecock_detector.py --test runs/detect/shuttlecock_train/weights/best.pt
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="badminton_ball_dataset/data.yaml",
        help="資料集設定檔（預設：badminton_ball_dataset/data.yaml）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="預訓練模型（預設：yolov8n.pt，也可用 yolov8s/m/l/x.pt）"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="訓練週期數（預設：50）"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批次大小（預設：16，M3/M4 可用 32）"
    )
    
    parser.add_argument(
        "--img",
        type=int,
        default=640,
        help="圖片大小（預設：640）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="運算裝置（預設：自動，可選 cpu/mps/0）"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="測試模型路徑（訓練後自動測試）"
    )
    
    parser.add_argument(
        "--test-video",
        type=str,
        default="pro/Chou-pro.mp4",
        help="測試影片路徑"
    )
    
    args = parser.parse_args()
    
    # 如果只是測試
    if args.test:
        test_detector(args.test, args.test_video)
        return
    
    # 訓練模型
    print("=" * 60)
    print("🏸 羽球偵測器訓練")
    print("=" * 60)
    
    model = train_shuttlecock_detector(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img,
        device=args.device
    )
    
    if model is not None:
        # 訓練成功，詢問是否測試
        print(f"\n" + "=" * 60)
        print(f"🎉 恭喜！你的羽球偵測器訓練完成！")
        print(f"=" * 60)
        print(f"\n💡 使用方法：")
        print(f"   1. 載入模型：")
        print(f"      model = YOLO('runs/detect/shuttlecock_train/weights/best.pt')")
        print(f"")
        print(f"   2. 偵測影片：")
        print(f"      results = model.predict('your_video.mp4', conf=0.25)")
        print(f"")
        print(f"   3. 測試此影片：")
        print(f"      python train_shuttlecock_detector.py --test runs/detect/shuttlecock_train/weights/best.pt")


if __name__ == "__main__":
    main()
