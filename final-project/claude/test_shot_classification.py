"""
測試擊球類型分類功能
可以用於調整閾值參數
"""

import torch
_original_load = torch.load
def _patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)
torch.load = _patched_load

import cv2
import numpy as np
from ultralytics import YOLO
from pose_analysis import (
    calculate_wrist_speed,
    calculate_body_lean,
    calculate_shoulder_rotation,
    calculate_body_side_angle,
    is_arm_raised,
    is_jumping,
    classify_shot_type,
    angle_3points
)

VIDEO_PATH = "20250711_short.mp4"
OUTPUT_PATH = "shot_classification_output.mp4"
POSE_MODEL_PATH = "yolo11n-pose.pt"

def main():
    print("=" * 60)
    print("擊球類型分類測試")
    print("=" * 60)
    
    # 載入模型
    print(f"載入模型: {POSE_MODEL_PATH}")
    try:
        pose_model = YOLO(POSE_MODEL_PATH)
    except:
        pose_model = YOLO("yolov8n-pose.pt")
    
    # 開啟影片
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 設定影片輸出
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    
    print(f"影片 FPS: {fps:.2f}, 總幀數: {total_frames}")
    print(f"解析度: {w}x{h}")
    print(f"輸出檔案: {OUTPUT_PATH}")
    print("\n開始分析...")
    print("=" * 60)
    
    frame_id = 0
    prev_keypoints = None
    shot_count = {'smash': 0, 'clear': 0, 'drop': 0, 'unknown': 0}
    shot_cooldown = 0
    current_shot_type = None
    wrist_speeds = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # 骨架偵測
        pose_results = pose_model(frame, imgsz=640, verbose=False)[0]
        
        if len(pose_results.keypoints) > 0:
            keypoints = pose_results.keypoints.xy[0].cpu().numpy()
            
            # 繪製骨架點
            for i, (x, y) in enumerate(keypoints):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            # 繪製骨架連線
            skeleton = [
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6],
                [5, 7], [7, 9], [6, 8], [8, 10],
                [0, 1], [0, 2], [1, 3], [2, 4],
            ]
            for connection in skeleton:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                    pt1 = keypoints[pt1_idx]
                    pt2 = keypoints[pt2_idx]
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(frame, 
                                (int(pt1[0]), int(pt1[1])),
                                (int(pt2[0]), int(pt2[1])),
                                (0, 255, 0), 2)
            
            if prev_keypoints is not None:
                # 計算手腕速度
                wrist_speed = calculate_wrist_speed(keypoints, prev_keypoints)
                wrist_speeds.append(wrist_speed)
                
                # 偵測擊球瞬間
                if shot_cooldown <= 0 and wrist_speed > 20:
                    # 檢查是否為局部峰值
                    if len(wrist_speeds) >= 3:
                        if wrist_speeds[-2] > wrist_speeds[-3] and wrist_speeds[-2] > wrist_speeds[-1]:
                            shot_type = classify_shot_type(prev_keypoints, keypoints)
                            
                            # 計算詳細特徵（用於調試）
                            elbow_angle = angle_3points(
                                keypoints[6], keypoints[8], keypoints[10]
                            )
                            body_lean = calculate_body_lean(keypoints)
                            jumping = is_jumping(keypoints)
                            shoulder_rot = calculate_shoulder_rotation(keypoints)
                            side_angle = calculate_body_side_angle(keypoints)
                            arm_raised = is_arm_raised(keypoints)
                            
                            print(f"\n📍 Frame {frame_id} (時間: {frame_id/fps:.2f}s)")
                            print(f"   擊球類型: {shot_type.upper()}")
                            print(f"   手腕速度: {wrist_speeds[-2]:.1f} px/frame")
                            print(f"   手肘角度: {elbow_angle:.1f}°")
                            print(f"   身體後仰: {body_lean:.1f}°")
                            print(f"   側身程度: {side_angle:.2f} (0=正面, 1=側面)")
                            print(f"   手臂抬起: {'✅ 是' if arm_raised else '❌ 否'}")
                            print(f"   跳躍狀態: {'是' if jumping else '否'}")
                            
                            shot_count[shot_type] += 1
                            current_shot_type = shot_type
                            shot_cooldown = 30  # 1秒冷卻
            
            prev_keypoints = keypoints.copy()
        
        # 顯示偵測結果
        if shot_cooldown > 0:
            shot_cooldown -= 1
            
            # 在冷卻期間顯示偵測結果（顯示約10幀）
            if shot_cooldown > 20 and current_shot_type:
                shot_color = {
                    'smash': (0, 0, 255),      # 紅色
                    'clear': (0, 255, 0),      # 綠色
                    'drop': (255, 0, 0),       # 藍色
                    'unknown': (128, 128, 128) # 灰色
                }.get(current_shot_type, (255, 255, 255))
                
                shot_text = {
                    'smash': 'SMASH (殺球)',
                    'clear': 'CLEAR (高遠球)',
                    'drop': 'DROP (切球)',
                    'unknown': 'UNKNOWN'
                }.get(current_shot_type, current_shot_type.upper())
                
                # 半透明背景
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//2 - 200, 50), (w//2 + 200, 150), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # 主要文字
                cv2.putText(frame, shot_text, 
                           (w//2 - 180, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, shot_color, 3)
        
        # 顯示幀數資訊
        cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 顯示即時狀態（如果有偵測到人）
        if len(pose_results.keypoints) > 0 and prev_keypoints is not None:
            side_angle = calculate_body_side_angle(keypoints)
            arm_raised = is_arm_raised(keypoints)
            
            status_y = h - 120
            cv2.putText(frame, f"Side: {side_angle:.2f}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 255) if side_angle > 0.3 else (128, 128, 128), 2)
            status_y += 30
            cv2.putText(frame, f"Arm: {'UP' if arm_raised else 'DOWN'}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if arm_raised else (128, 128, 128), 2)
        
        # 顯示統計資訊
        y_offset = 70
        cv2.putText(frame, f"Smash: {shot_count['smash']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Clear: {shot_count['clear']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame, f"Drop: {shot_count['drop']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 寫入影片
        out.write(frame)
        
        # 顯示進度
        if frame_id % 100 == 0:
            progress = frame_id / total_frames * 100
            print(f"進度: {frame_id}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    print(f"偵測到的擊球統計:")
    print(f"  殺球 (Smash): {shot_count['smash']}")
    print(f"  高遠球 (Clear): {shot_count['clear']}")
    print(f"  切球 (Drop): {shot_count['drop']}")
    print(f"  未分類: {shot_count['unknown']}")
    print(f"  總計: {sum(shot_count.values())}")
    print(f"\n輸出影片: {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
