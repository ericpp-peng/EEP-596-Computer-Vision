"""
視覺化羽球偵測結果
生成帶有偵測標記的影片,方便檢查偵測效果
"""

import cv2
import sys
import torch

# 修正 PyTorch 2.6 問題
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from shuttlecock_detector import ShuttlecockDetector
from ultralytics import YOLO

def main():
    VIDEO_PATH = "20250711_short.mp4"
    OUTPUT_PATH = "detection_test_output.mp4"
    MAX_FRAMES = 300  # 只處理前 5 秒 (300幀 @ 60fps)
    
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    
    print("=" * 60)
    print("羽球偵測視覺化測試")
    print("=" * 60)
    print(f"輸入影片: {VIDEO_PATH}")
    print(f"輸出影片: {OUTPUT_PATH}")
    print(f"處理幀數: {MAX_FRAMES}")
    print("=" * 60)
    
    # 初始化
    print("\n初始化偵測器 (啟用場地範圍限制)...")
    detector = ShuttlecockDetector(mode='hybrid', court_area='auto')
    pose_model = YOLO("yolov8n-pose.pt")
    
    # 開啟影片
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 輸出影片
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    
    print(f"\n開始處理 (解析度: {w}x{h} @ {fps:.1f} fps)...\n")
    
    stats = {
        'total': 0,
        'detected': 0,
        'yolo': 0,
        'color': 0,
        'wrist': 0
    }
    
    for i in range(MAX_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        
        stats['total'] += 1
        
        # Pose 偵測 (用於手腕估計)
        pose_results = pose_model(frame, imgsz=640)[0]
        keypoints = None
        
        if len(pose_results.keypoints) > 0:
            keypoints = pose_results.keypoints.xy[0].cpu().numpy()
            
            # 畫骨架 (半透明)
            for j, (x, y) in enumerate(keypoints):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # 羽球偵測
        result = detector.detect(frame, keypoints)
        
        if result:
            stats['detected'] += 1
            source = result.get('source', 'detect')
            
            if source == 'wrist':
                stats['wrist'] += 1
            else:
                # 根據信心度判斷來源
                conf = result.get('conf', 0)
                if conf > 0.6:
                    stats['yolo'] += 1
                else:
                    stats['color'] += 1
            
            # 視覺化
            detector.draw(frame, result)
        
        # 顯示統計資訊
        detection_rate = (stats['detected'] / stats['total']) * 100
        
        info_y = 30
        cv2.putText(frame, f"Frame: {i+1}/{MAX_FRAMES}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 35
        cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 35
        if result:
            source = result.get('source', 'detect')
            conf = result.get('conf', 0)
            cv2.putText(frame, f"Source: {source} (conf={conf:.2f})", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 寫入輸出
        out.write(frame)
        
        # 進度顯示
        if (i + 1) % 50 == 0:
            print(f"  處理進度: {i+1}/{MAX_FRAMES} ({(i+1)/MAX_FRAMES*100:.1f}%), 偵測率: {detection_rate:.1f}%")
    
    cap.release()
    out.release()
    
    # 顯示最終統計
    print("\n" + "=" * 60)
    print("處理完成!")
    print("=" * 60)
    print(f"總幀數: {stats['total']}")
    print(f"偵測成功: {stats['detected']} ({stats['detected']/stats['total']*100:.1f}%)")
    print(f"  - YOLO 偵測: {stats['yolo']}")
    print(f"  - 顏色偵測: {stats['color']}")
    print(f"  - 手腕估計: {stats['wrist']}")
    print(f"\n✅ 輸出影片: {OUTPUT_PATH}")
    print(f"\n使用以下指令播放:")
    print(f"  open {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
