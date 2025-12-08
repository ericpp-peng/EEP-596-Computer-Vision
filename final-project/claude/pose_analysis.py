"""
羽球選手姿勢分析系統 - 簡化版
專注於骨架偵測、角度分析、專業對比

功能:
1. 人物骨架偵測 (YOLO11-pose)
2. 球場線條繪製
3. 關鍵角度計算
4. 專業姿勢對比與評分

作者: Final Project
日期: 2025/12/08
"""

# === PyTorch 2.6 相容性修正 ===
import torch
_original_load = torch.load
def _patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)
torch.load = _patched_load

import cv2
import numpy as np
import pickle
import os
import time
from ultralytics import YOLO

# =============================
# 配置參數
# =============================

VIDEO_PATH = "20250711_short.mp4"
OUTPUT_PATH = "pose_analysis_output.mp4"
CORNERS_FILE = "court_corners.pkl"

# YOLO 模型
POSE_MODEL_PATH = "yolo11n-pose.pt"  # 或 "yolov8n-pose.pt"

# 專業標準 (以殺球為例)
PRO_STANDARDS = {
    "shoulder_angle": 145,  # 肩膀角度 (度)
    "elbow_angle": 165,     # 手肘角度 (度)
    "hip_angle": 170,       # 髖關節角度 (度)
}

# 全域變數
COURT_CORNERS = None


# =============================
# 幾何計算函數
# =============================

def distance(p1, p2):
    """計算兩點距離"""
    if p1 is None or p2 is None:
        return float('inf')
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def angle_3points(p1, p2, p3):
    """
    計算三點形成的角度 (以 p2 為頂點)
    返回角度 (度)
    """
    if p1 is None or p2 is None or p3 is None:
        return 0
    
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


# =============================
# 視覺化函數
# =============================

def draw_court_lines(frame, corners):
    """
    在畫面上繪製球場線條
    corners: numpy array, shape (4, 2)
    """
    if corners is None or len(corners) != 4:
        return frame
    
    corners = corners.astype(np.int32)
    
    # 外框
    cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 255), thickness=3)
    
    # 中線 (左右對分)
    mid_top = ((corners[0] + corners[1]) // 2).tolist()
    mid_bot = ((corners[3] + corners[2]) // 2).tolist()
    cv2.line(frame, mid_top, mid_bot, (0, 255, 255), 2)
    
    # 標註角點
    for i, corner in enumerate(corners):
        cv2.circle(frame, tuple(corner), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"P{i+1}", (corner[0]+10, corner[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def draw_skeleton(frame, keypoints):
    """
    繪製骨架連線
    keypoints: (17, 2) numpy array
    """
    # COCO skeleton connections
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # 下半身
        [5, 11], [6, 12],  # 軀幹
        [5, 6],  # 肩膀
        [5, 7], [7, 9],  # 左手臂
        [6, 8], [8, 10],  # 右手臂
        [0, 1], [0, 2],  # 臉部
        [1, 3], [2, 4],
    ]
    
    for connection in skeleton:
        pt1_idx, pt2_idx = connection
        if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            # 只畫有效點
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(frame, 
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        (0, 255, 0), 2)


def calculate_angles(keypoints):
    """
    計算關鍵角度
    返回: dict of angles
    """
    if keypoints is None or len(keypoints) < 17:
        return None
    
    # COCO Keypoints 索引
    # 5: L_Shoulder, 6: R_Shoulder
    # 7: L_Elbow, 8: R_Elbow
    # 9: L_Wrist, 10: R_Wrist
    # 11: L_Hip, 12: R_Hip
    
    r_shoulder = keypoints[6]
    r_elbow = keypoints[8]
    r_wrist = keypoints[10]
    r_hip = keypoints[12]
    l_hip = keypoints[11]
    
    # 計算角度
    angles = {
        "shoulder": angle_3points(r_elbow, r_shoulder, r_hip),
        "elbow": angle_3points(r_shoulder, r_elbow, r_wrist),
        "hip": angle_3points(r_shoulder, r_hip, l_hip),
    }
    
    return angles


def compare_with_standard(angles):
    """
    與專業標準比較
    返回: feedback dict
    """
    if angles is None:
        return None
    
    feedback = {
        "score": 100,
        "suggestions": []
    }
    
    # 肩膀角度檢查
    shoulder_diff = angles["shoulder"] - PRO_STANDARDS["shoulder_angle"]
    if abs(shoulder_diff) > 15:
        feedback["score"] -= 30
        if shoulder_diff < 0:
            feedback["suggestions"].append(f"肩膀抬高不足 ({shoulder_diff:.1f}°)")
        else:
            feedback["suggestions"].append(f"肩膀過度抬高 (+{shoulder_diff:.1f}°)")
    
    # 手肘角度檢查
    elbow_diff = angles["elbow"] - PRO_STANDARDS["elbow_angle"]
    if abs(elbow_diff) > 20:
        feedback["score"] -= 25
        if elbow_diff < 0:
            feedback["suggestions"].append(f"手臂未充分伸直 ({elbow_diff:.1f}°)")
        else:
            feedback["suggestions"].append(f"手臂過度僵硬 (+{elbow_diff:.1f}°)")
    
    # 髖關節角度檢查
    hip_diff = angles["hip"] - PRO_STANDARDS["hip_angle"]
    if abs(hip_diff) > 15:
        feedback["score"] -= 20
        if hip_diff < 0:
            feedback["suggestions"].append(f"髖部過度彎曲 ({hip_diff:.1f}°)")
    
    if len(feedback["suggestions"]) == 0:
        feedback["suggestions"].append("✅ 姿勢標準!")
    
    feedback["score"] = max(0, feedback["score"])
    
    return feedback


def draw_analysis_overlay(frame, angles, feedback):
    """
    繪製分析資訊覆蓋層
    """
    if angles is None:
        return
    
    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 80), (400, 350), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # 標題
    cv2.putText(frame, "Pose Analysis", (20, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # 角度資訊
    y_offset = 145
    cv2.putText(frame, f"Shoulder: {angles['shoulder']:.1f}° (Std: {PRO_STANDARDS['shoulder_angle']}°)",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(frame, f"Elbow: {angles['elbow']:.1f}° (Std: {PRO_STANDARDS['elbow_angle']}°)",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(frame, f"Hip: {angles['hip']:.1f}° (Std: {PRO_STANDARDS['hip_angle']}°)",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 評分
    if feedback:
        y_offset += 40
        score_color = (0, 255, 0) if feedback["score"] >= 80 else (0, 165, 255) if feedback["score"] >= 60 else (0, 0, 255)
        cv2.putText(frame, f"Score: {feedback['score']}/100",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        # 建議
        y_offset += 35
        for i, suggestion in enumerate(feedback["suggestions"][:3]):  # 最多顯示3條
            cv2.putText(frame, suggestion,
                       (20, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)


# =============================
# 主程式
# =============================

def main():
    global COURT_CORNERS
    
    print("=" * 60)
    print("羽球選手姿勢分析系統 - 簡化版")
    print("=" * 60)
    
    # 載入球場角點 (可選)
    if os.path.exists(CORNERS_FILE):
        with open(CORNERS_FILE, 'rb') as f:
            COURT_CORNERS = pickle.load(f)
        print(f"✅ 已載入球場角點: {CORNERS_FILE}")
    else:
        print(f"ℹ️  未找到球場角點檔案，將跳過球場線條繪製")
        print(f"   (可執行 court_calibration.py 進行標定)")
    
    # 載入模型
    print(f"\n載入 YOLO Pose 模型: {POSE_MODEL_PATH}")
    try:
        pose_model = YOLO(POSE_MODEL_PATH)
        print("✅ 模型載入成功")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        print("   嘗試使用 yolov8n-pose.pt")
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
    
    print(f"\n影片資訊:")
    print(f"  解析度: {w}x{h}")
    print(f"  FPS: {fps:.2f}")
    print(f"  總幀數: {total_frames}")
    
    # 輸出影片
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    
    print(f"\n開始處理...")
    print("=" * 60)
    
    frame_id = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # === 1. 骨架偵測 ===
        pose_results = pose_model(frame, imgsz=640, verbose=False)[0]
        
        keypoints = None
        if len(pose_results.keypoints) > 0:
            # 取第一個人的骨架
            kpts = pose_results.keypoints.xy[0].cpu().numpy()  # (17, 2)
            keypoints = kpts
            
            # 繪製骨架點
            for i, (x, y) in enumerate(kpts):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            # 繪製骨架連線
            draw_skeleton(frame, kpts)
        
        # === 2. 球場線條 ===
        if COURT_CORNERS is not None:
            frame = draw_court_lines(frame, COURT_CORNERS)
        
        # === 3. 角度分析 ===
        angles = None
        feedback = None
        
        if keypoints is not None:
            angles = calculate_angles(keypoints)
            if angles:
                feedback = compare_with_standard(angles)
                draw_analysis_overlay(frame, angles, feedback)
        
        # === 4. 顯示資訊 ===
        cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 寫入輸出
        out.write(frame)
        
        # 顯示進度
        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_id / elapsed
            eta = (total_frames - frame_id) / fps_current
            print(f"進度: {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%) | "
                  f"處理速度: {fps_current:.1f} fps | ETA: {eta:.1f}s")
    
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    
    print("=" * 60)
    print("✅ 處理完成!")
    print(f"   輸出檔案: {OUTPUT_PATH}")
    print(f"   處理時間: {total_time:.1f} 秒")
    print(f"   處理速度: {total_frames/total_time:.1f} fps")
    print("=" * 60)


if __name__ == "__main__":
    main()
