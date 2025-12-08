"""
羽球動作分析系統 - MVP Version
功能:
1. YOLO11 pose 偵測人物骨架
2. 偵測羽球、球拍
3. 球場線條繪製
4. 擊球瞬間偵測
5. 動作分類與角度分析
6. 與專業姿勢比較
"""

import os
import pickle
from collections import deque
import time
import numpy as np
import cv2

# 修正 PyTorch 2.6 weights_only 問題
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO
from shuttlecock_detector import ShuttlecockDetector, RacketDetector# =============================
# 配置參數
# =============================
VIDEO_PATH = "20250711_short.mp4"
OUTPUT_PATH = "badminton_analysis_output.mp4"

# YOLO 模型 (可以用 yolo8 或 yolo11)
POSE_MODEL_PATH = "yolo11n-pose.pt"  # 如果沒有就用 "yolov8n-pose.pt"
DET_MODEL_PATH = "yolov8n.pt"  # 用於偵測球、球拍

# 羽球偵測模式: 'yolo', 'color', 'hybrid', 'custom'
# hybrid = YOLO + 顏色偵測 (最推薦)
SHUTTLECOCK_MODE = "hybrid"
CUSTOM_MODEL_PATH = None  # 如果有自訓練模型, 填路徑

# 球場四個角點 (從 court_corners.pkl 讀取)
CORNERS_FILE = "court_corners.pkl"
COURT_CORNERS = None

# 專業標準姿勢參數 (以殺球為例)
PRO_SMASH = {
    "shoulder_angle": 85,      # 肩膀角度
    "elbow_angle": 160,        # 肘部角度 (接近伸直)
    "wrist_snap_speed": 20,    # 手腕速度
    "shoulder_rotation": 45,   # 肩膀旋轉
}

# =============================
# 工具函式
# =============================

def angle_3points(a, b, c):
    """計算三點形成的角度 (度數)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def distance(p1, p2):
    """計算兩點距離"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def draw_court_lines(frame, corners):
    """繪製球場線條 (藍色)"""
    if corners is None:
        return frame
    
    corners = np.array(corners, dtype=np.int32)
    # 外框
    cv2.polylines(frame, [corners], isClosed=True, color=(255, 100, 0), thickness=2)
    
    # 中線 (左右對分)
    mid_top = ((corners[0] + corners[1]) // 2).tolist()
    mid_bot = ((corners[3] + corners[2]) // 2).tolist()
    cv2.line(frame, mid_top, mid_bot, (255, 100, 0), 2)
    
    return frame


def detect_impact(ball_pos, ball_history, racket_pos, threshold_dist=50, threshold_speed=15):
    """
    偵測擊球瞬間
    條件: 
    1. 球與球拍距離很近
    2. 球速突然變大
    """
    if ball_pos is None or racket_pos is None:
        return False
    
    # 距離檢查
    dist = distance(ball_pos, racket_pos)
    if dist > threshold_dist:
        return False
    
    # 球速檢查 (與前一幀比較)
    if len(ball_history) >= 2:
        speed = distance(ball_history[-1], ball_history[-2])
        if speed > threshold_speed:
            return True
    
    return False


def classify_action(keypoints, ball_velocity):
    """
    動作分類 (rule-based)
    keypoints: 17個關鍵點 [[x,y], ...]
    ball_velocity: (vx, vy)
    """
    # COCO pose keypoints index:
    # 6: R_shoulder, 8: R_elbow, 10: R_wrist, 12: R_hip
    
    if keypoints is None or len(keypoints) < 17:
        return "UNKNOWN", {}
    
    r_shoulder = keypoints[6]
    r_elbow = keypoints[8]
    r_wrist = keypoints[10]
    r_hip = keypoints[12]
    
    # 計算角度
    elbow_angle = angle_3points(r_shoulder, r_elbow, r_wrist)
    shoulder_angle = angle_3points(r_elbow, r_shoulder, r_hip)
    
    angles = {
        "shoulder": shoulder_angle,
        "elbow": elbow_angle,
    }
    
    # 分類邏輯
    ball_speed = np.linalg.norm(ball_velocity)
    ball_dir_down = ball_velocity[1] > 0  # y 方向向下
    
    if ball_dir_down and ball_speed > 20:
        return "SMASH", angles
    elif ball_dir_down and ball_speed < 15:
        return "DROP", angles
    elif not ball_dir_down and ball_speed > 15:
        return "CLEAR", angles
    else:
        return "OTHER", angles


def compare_with_pro(action_type, angles):
    """
    與專業姿勢比較
    輸出建議
    """
    if action_type != "SMASH":
        return None  # 目前只有 SMASH 的標準
    
    shoulder_diff = angles["shoulder"] - PRO_SMASH["shoulder_angle"]
    elbow_diff = angles["elbow"] - PRO_SMASH["elbow_angle"]
    
    feedback = {
        "score": 0,
        "suggestions": []
    }
    
    # 評分 (100分制)
    score = 100
    
    if abs(shoulder_diff) > 15:
        score -= 30
        if shoulder_diff < 0:
            feedback["suggestions"].append(f"肩膀抬高不足 ({shoulder_diff:.1f}°)")
        else:
            feedback["suggestions"].append(f"肩膀過度抬高 (+{shoulder_diff:.1f}°)")
    
    if abs(elbow_diff) > 20:
        score -= 25
        if elbow_diff < 0:
            feedback["suggestions"].append(f"手臂未充分伸直 ({elbow_diff:.1f}°)")
        else:
            feedback["suggestions"].append(f"手臂過度僵硬")
    
    if len(feedback["suggestions"]) == 0:
        feedback["suggestions"].append("動作標準!")
    
    feedback["score"] = max(0, score)
    return feedback


# =============================
# 主程式
# =============================

def main():
    global COURT_CORNERS
    
    # 載入球場角點
    if os.path.exists(CORNERS_FILE):
        with open(CORNERS_FILE, 'rb') as f:
            COURT_CORNERS = pickle.load(f)
        print(f"✅ 已載入球場角點: {CORNERS_FILE}")
    else:
        print(f"⚠️  未找到 {CORNERS_FILE}, 請先執行 court_calibration.py")
    
    # 載入模型
    print("載入模型...")
    try:
        pose_model = YOLO(POSE_MODEL_PATH)
    except:
        print(f"找不到 {POSE_MODEL_PATH}, 嘗試使用 yolov8n-pose.pt")
        pose_model = YOLO("yolov8n-pose.pt")
    
    det_model = YOLO(DET_MODEL_PATH)
    
    # 初始化偵測器 (啟用自動場地範圍限制)
    print(f"初始化羽球偵測器 (模式: {SHUTTLECOCK_MODE})...")
    shuttlecock_detector = ShuttlecockDetector(
        mode=SHUTTLECOCK_MODE,
        model_path=DET_MODEL_PATH,
        custom_model_path=CUSTOM_MODEL_PATH,
        court_area='auto'  # 自動使用畫面中心 70% 區域
    )
    racket_detector = RacketDetector(mode='wrist')
    
    # 開啟影片
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"無法開啟影片: {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 輸出影片
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    
    print(f"開始處理影片: {total_frames} frames, {fps} fps")
    
    # 歷史資料
    keypoints_prev = None
    
    frame_id = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # === 1. Pose 偵測 ===
        pose_results = pose_model(frame, imgsz=640)[0]
        
        keypoints = None
        if len(pose_results.keypoints) > 0:
            # 取第一個人的骨架 (或選最大的)
            kpts = pose_results.keypoints.xy[0].cpu().numpy()  # shape: (17, 2)
            keypoints = kpts
            
            # 畫骨架
            for i, (x, y) in enumerate(kpts):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        # === 2. 偵測球/球拍 (使用進階偵測器) ===
        ball_detection = shuttlecock_detector.detect(frame, keypoints)
        racket_detection = racket_detector.detect(frame, keypoints, det_model)
        
        ball_pos = ball_detection['pos'] if ball_detection else None
        racket_pos = racket_detection['pos'] if racket_detection else None
        
        # 視覺化羽球偵測
        if ball_detection:
            shuttlecock_detector.draw(frame, ball_detection)
        
        # === 3. 球場線條 ===
        if COURT_CORNERS is not None:
            frame = draw_court_lines(frame, COURT_CORNERS)
        
        # === 4. 擊球偵測 ===
        # 使用偵測器的歷史資料
        ball_history = list(shuttlecock_detector.history)
        
        is_impact = detect_impact(ball_pos, ball_history, racket_pos)
        
        # === 5. 動作分類 ===
        action_type = "N/A"
        feedback = None
        
        if is_impact and keypoints is not None:
            # 計算球速向量
            ball_vel = shuttlecock_detector.get_velocity()
            
            action_type, angles = classify_action(keypoints, ball_vel)
            
            # 與 Pro 比較
            feedback = compare_with_pro(action_type, angles)
            
            # 顯示黃色警告
            cv2.putText(frame, f"IMPACT! {action_type}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            if feedback:
                y_offset = 150
                cv2.putText(frame, f"Score: {feedback['score']}/100", (50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 35
                for suggestion in feedback['suggestions']:
                    cv2.putText(frame, suggestion, (50, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    y_offset += 30
        
        # === 6. 顯示資訊 ===
        cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 寫入輸出
        out.write(frame)
        
        # 每 100 frames 顯示進度
        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / frame_id) * (total_frames - frame_id)
            print(f"處理進度: {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%), ETA: {eta:.1f}s")
        
        keypoints_prev = keypoints
    
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print(f"\n✅ 完成! 輸出: {OUTPUT_PATH}")
    print(f"處理時間: {total_time:.1f} 秒")
    print(f"處理速度: {total_frames/total_time:.1f} fps")


if __name__ == "__main__":
    # 提示: 你需要先設定球場角點
    print("=" * 60)
    print("羽球動作分析系統 MVP")
    print("=" * 60)
    print("\n注意事項:")
    print("1. 請確認影片路徑: 20250711_short.mp4")
    print("2. 如需球場線條, 請先執行球場標定 (參考 court_calibration.py)")
    print("3. 預設使用 COCO sports ball (class 32), 建議訓練羽球偵測模型")
    print("\n按 Ctrl+C 中斷\n")
    
    main()
