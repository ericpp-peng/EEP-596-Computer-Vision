"""
簡單的羽球偵測測試
不使用視窗,只輸出偵測結果
"""

import cv2
import sys
import torch

# 修正 PyTorch 2.6 問題 - 必須在最前面
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from shuttlecock_detector import ShuttlecockDetector

def main():
    VIDEO_PATH = "20250711_short.mp4"
    
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    
    print("=" * 60)
    print("羽球偵測器簡單測試")
    print("=" * 60)
    
    # 初始化偵測器
    print("\n1. 初始化偵測器 (Hybrid 模式)...")
    detector = ShuttlecockDetector(mode='hybrid')
    print("   ✅ 完成")
    
    # 開啟影片
    print(f"\n2. 開啟影片: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"   ❌ 無法開啟影片")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   ✅ 影片資訊: {total_frames} 幀 @ {fps:.2f} fps")
    
    # 測試前 100 幀
    print(f"\n3. 測試偵測 (前 100 幀)...")
    detected_count = 0
    test_frames = min(100, total_frames)
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame)
        if result:
            detected_count += 1
            
            # 每 20 幀顯示一次
            if i % 20 == 0:
                pos = result['pos']
                conf = result.get('conf', 0)
                source = result.get('source', 'detect')
                print(f"   幀 {i:3d}: 偵測到羽球 @ ({pos[0]:.0f}, {pos[1]:.0f}), 信心度={conf:.2f}, 來源={source}")
    
    cap.release()
    
    # 統計
    detection_rate = (detected_count / test_frames) * 100
    print(f"\n4. 偵測統計:")
    print(f"   測試幀數: {test_frames}")
    print(f"   偵測成功: {detected_count} ({detection_rate:.1f}%)")
    
    if detection_rate > 70:
        print(f"   ✅ 偵測率良好!")
    elif detection_rate > 40:
        print(f"   ⚠️  偵測率中等,可能需要調整參數")
    else:
        print(f"   ❌ 偵測率較低,建議使用手腕估計或訓練自訂模型")
    
    print("\n" + "=" * 60)
    print("測試完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
