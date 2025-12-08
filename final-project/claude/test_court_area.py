"""
場地範圍限制測試
只偵測指定場地內的羽球
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
    OUTPUT_PATH = "court_limited_detection.mp4"
    MAX_FRAMES = 300
    
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    
    print("=" * 60)
    print("場地範圍限制測試")
    print("=" * 60)
    print(f"輸入影片: {VIDEO_PATH}")
    print(f"輸出影片: {OUTPUT_PATH}")
    print(f"處理幀數: {MAX_FRAMES}")
    print("=" * 60)
    
    # 初始化偵測器 - 使用自動場地範圍
    print("\n初始化偵測器 (自動場地範圍)...")
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
    
    print(f"\n開始處理 (解析度: {w}x{h} @ {fps:.1f} fps)...")
    print("綠色半透明區域 = 場地範圍\n")
    
    stats = {
        'total': 0,
        'detected': 0,
        'in_court': 0,
        'out_court': 0
    }
    
    for i in range(MAX_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        
        stats['total'] += 1
        
        # Pose 偵測
        pose_results = pose_model(frame, imgsz=640)[0]
        keypoints = None
        
        if len(pose_results.keypoints) > 0:
            keypoints = pose_results.keypoints.xy[0].cpu().numpy()
            
            # 畫骨架
            for j, (x, y) in enumerate(keypoints):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # 羽球偵測 (會自動過濾場地外的偵測)
        result = detector.detect(frame, keypoints)
        
        if result:
            stats['detected'] += 1
            stats['in_court'] += 1
        
        # 視覺化 (顯示場地範圍)
        detector.draw(frame, result, show_court=True)
        
        # 顯示統計資訊
        detection_rate = (stats['detected'] / stats['total']) * 100
        
        info_y = 30
        cv2.putText(frame, f"Frame: {i+1}/{MAX_FRAMES}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 35
        cv2.putText(frame, f"In-Court Detection: {detection_rate:.1f}%", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 35
        cv2.putText(frame, f"Court Area: Auto (Center 70%)", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
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
    print(f"場地內偵測: {stats['detected']} ({stats['detected']/stats['total']*100:.1f}%)")
    print(f"\n✅ 輸出影片: {OUTPUT_PATH}")
    print(f"\n使用以下指令播放:")
    print(f"  open {OUTPUT_PATH}")
    print("\n💡 綠色半透明區域顯示場地範圍")
    print("   只有在此區域內的羽球會被偵測")
    print("=" * 60)

if __name__ == "__main__":
    main()
