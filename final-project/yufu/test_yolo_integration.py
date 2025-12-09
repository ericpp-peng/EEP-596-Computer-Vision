"""
測試 YOLO 羽球偵測整合
"""
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# 修正 PyTorch weights_only 問題
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

print("=" * 60)
print("測試 YOLO 羽球偵測整合")
print("=" * 60)

# 設定 GPU
if torch.backends.mps.is_available():
    device = 'mps'
    print("✅ 使用 MPS (Metal) GPU 加速")
else:
    device = 'cpu'
    print("⚠️  使用 CPU")

# 載入模型
print("\n載入模型...")
pose_model = YOLO("yolov8n-pose.pt")
pose_model.to(device)
print("✅ 姿態模型已載入")

shuttlecock_model = YOLO("runs/detect/shuttlecock_train/weights/best.pt")
shuttlecock_model.to(device)
print("✅ 羽球偵測模型已載入")

# 測試影片
VIDEO_PATH = "20250711_short.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ 無法開啟影片: {VIDEO_PATH}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n影片資訊: {w}x{h} @ {fps} FPS")

# 處理前 100 幀
print("\n開始處理...")
print("按 'q' 退出, 空白鍵暫停")

frame_count = 0
detection_count = 0

while frame_count < 100:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # YOLO 羽球偵測
    results = shuttlecock_model(
        frame,
        conf=0.25,
        imgsz=640,
        verbose=False,
        device=device,
        half=True if device == 'mps' else False
    )[0]
    
    # 標示偵測結果
    if len(results.boxes) > 0:
        detection_count += 1
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            # 畫框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 顯示置信度
            label = f"Ball {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 顯示幀數和偵測統計
    info = f"Frame: {frame_count}/100  Detected: {detection_count}"
    cv2.putText(frame, info, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 顯示畫面
    cv2.imshow("YOLO Shuttlecock Detection Test", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

# 統計結果
print("\n" + "=" * 60)
print("測試完成")
print("=" * 60)
print(f"處理幀數: {frame_count}")
print(f"偵測到羽球: {detection_count} 幀")
print(f"偵測率: {detection_count/frame_count*100:.1f}%")
print("=" * 60)
