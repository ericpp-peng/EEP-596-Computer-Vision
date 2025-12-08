"""
快速測試羽球偵測器
測試不同模式的偵測效果
"""

import cv2
import sys
import os

# 修正 PyTorch 2.6 weights_only 問題
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from shuttlecock_detector import ShuttlecockDetector, RacketDetector
from ultralytics import YOLO

def test_detection_modes():
    """測試不同偵測模式"""
    
    VIDEO_PATH = "20250711_short.mp4"
    
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("=" * 60)
    print("羽球偵測器測試")
    print("=" * 60)
    print(f"影片: {VIDEO_PATH}")
    print(f"解析度: {w}x{h} @ {fps} fps")
    print()
    print("按鍵控制:")
    print("  1: YOLO 模式")
    print("  2: 顏色偵測模式")
    print("  3: Hybrid 模式 (推薦)")
    print("  space: 暫停")
    print("  q: 退出")
    print("=" * 60)
    
    # 載入 pose 模型 (用於手腕估計)
    pose_model = YOLO("yolov8n-pose.pt")
    
    # 初始偵測器 (hybrid 模式)
    detector = ShuttlecockDetector(mode='hybrid')
    racket_detector = RacketDetector(mode='wrist')
    
    current_mode = 'hybrid'
    
    detection_stats = {
        'total_frames': 0,
        'detected_frames': 0,
        'yolo_count': 0,
        'color_count': 0,
        'wrist_count': 0
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detection_stats['total_frames'] += 1
        
        # Pose 偵測 (取得手腕位置)
        pose_results = pose_model(frame, imgsz=640)[0]
        keypoints = None
        
        if len(pose_results.keypoints) > 0:
            keypoints = pose_results.keypoints.xy[0].cpu().numpy()
            
            # 畫骨架
            for i, (x, y) in enumerate(keypoints):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # 羽球偵測
        ball_detection = detector.detect(frame, keypoints)
        
        # 統計
        if ball_detection:
            detection_stats['detected_frames'] += 1
            source = ball_detection.get('source', 'detect')
            if source == 'wrist':
                detection_stats['wrist_count'] += 1
            elif current_mode == 'yolo':
                detection_stats['yolo_count'] += 1
            elif current_mode == 'color':
                detection_stats['color_count'] += 1
            else:
                # hybrid 可能混合
                if ball_detection.get('conf', 0) > 0.5:
                    detection_stats['yolo_count'] += 1
                else:
                    detection_stats['color_count'] += 1
        
        # 視覺化
        detector.draw(frame, ball_detection)
        
        # 顯示資訊
        cv2.putText(frame, f"Mode: {current_mode.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        detection_rate = detection_stats['detected_frames'] / max(detection_stats['total_frames'], 1) * 100
        cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Frame: {detection_stats['total_frames']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 顯示
        cv2.imshow("Shuttlecock Detection Test", frame)
        
        # 按鍵處理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        elif key == ord('1'):
            current_mode = 'yolo'
            detector = ShuttlecockDetector(mode='yolo')
            print("切換到 YOLO 模式")
        elif key == ord('2'):
            current_mode = 'color'
            detector = ShuttlecockDetector(mode='color')
            print("切換到顏色偵測模式")
        elif key == ord('3'):
            current_mode = 'hybrid'
            detector = ShuttlecockDetector(mode='hybrid')
            print("切換到 Hybrid 模式")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 顯示統計
    print("\n" + "=" * 60)
    print("偵測統計")
    print("=" * 60)
    print(f"總幀數: {detection_stats['total_frames']}")
    print(f"偵測成功: {detection_stats['detected_frames']} ({detection_rate:.1f}%)")
    print(f"  - YOLO 偵測: {detection_stats['yolo_count']}")
    print(f"  - 顏色偵測: {detection_stats['color_count']}")
    print(f"  - 手腕估計: {detection_stats['wrist_count']}")
    print("=" * 60)


if __name__ == "__main__":
    test_detection_modes()
