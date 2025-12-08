"""
羽球偵測模型訓練腳本 (選用)
如果時間允許,訓練專門的羽球/球拍偵測器

使用前準備:
1. 標記 50-100 張圖片 (用 Roboflow 或 LabelImg)
2. 準備 badminton.yaml:
   
   path: ./badminton_dataset
   train: images/train
   val: images/val
   
   names:
     0: shuttlecock
     1: racket

3. 執行此腳本
"""

from ultralytics import YOLO
import torch

# === 配置 ===
MODEL_BASE = "yolo11n.pt"  # 或 "yolov8n.pt"
DATA_YAML = "badminton.yaml"
EPOCHS = 100
BATCH = 16
IMG_SIZE = 640

# 訓練參數
PATIENCE = 20  # early stopping
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("羽球偵測模型訓練")
print("=" * 60)
print(f"基礎模型: {MODEL_BASE}")
print(f"資料集: {DATA_YAML}")
print(f"裝置: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print("=" * 60)

# === 載入模型 ===
model = YOLO(MODEL_BASE)

# === 訓練 ===
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMG_SIZE,
    device=DEVICE,
    patience=PATIENCE,
    
    # 優化參數
    augment=True,
    mosaic=1.0,      # 資料增強
    copy_paste=0.3,
    mixup=0.1,
    
    # 小物體優化
    scale=0.5,       # 縮放範圍
    fliplr=0.5,      # 左右翻轉
    
    # 其他
    workers=4,
    project="runs/detect",
    name="badminton",
    exist_ok=True,
    
    # 提早停止
    patience=20,
)

print("\n" + "=" * 60)
print("✅ 訓練完成!")
print(f"最佳權重: runs/detect/badminton/weights/best.pt")
print("=" * 60)

# === 驗證 ===
print("\n開始驗證...")
metrics = model.val()

print(f"\nmAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# === 測試 ===
print("\n測試推理...")
test_img = "test_image.jpg"  # 換成你的測試圖片

try:
    test_results = model(test_img)
    test_results[0].save("test_prediction.jpg")
    print(f"✅ 測試結果已儲存: test_prediction.jpg")
except Exception as e:
    print(f"測試失敗: {e}")

print("\n" + "=" * 60)
print("使用方式:")
print("  from ultralytics import YOLO")
print("  model = YOLO('runs/detect/badminton/weights/best.pt')")
print("  results = model('your_video.mp4')")
print("=" * 60)
