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


def calculate_wrist_speed(keypoints, prev_keypoints):
    """
    計算手腕速度（用於偵測擊球瞬間）
    返回: 速度值 (pixels/frame)
    """
    if keypoints is None or prev_keypoints is None:
        return 0
    
    if len(keypoints) < 17 or len(prev_keypoints) < 17:
        return 0
    
    # 右手腕 (index 10)
    wrist = keypoints[10]
    prev_wrist = prev_keypoints[10]
    
    if wrist[0] <= 0 or prev_wrist[0] <= 0:
        return 0
    
    speed = np.linalg.norm(wrist - prev_wrist)
    return speed


def calculate_body_lean(keypoints):
    """
    計算身體後仰角度（肩-髖-腳踝）
    返回: 後仰角度 (度)，0度 = 完全直立
    """
    if keypoints is None or len(keypoints) < 17:
        return 0
    
    shoulder = keypoints[6]  # R_Shoulder
    hip = keypoints[12]      # R_Hip
    ankle = keypoints[16]    # R_Ankle
    
    # 檢查關鍵點有效性
    if shoulder[0] <= 0 or hip[0] <= 0 or ankle[0] <= 0:
        return 0
    
    # 計算角度
    vector1 = shoulder - hip
    vector2 = ankle - hip
    
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.degrees(np.arccos(cos_angle))
    
    # 返回偏離直立的角度
    return abs(180 - angle)


def calculate_shoulder_rotation(keypoints):
    """
    計算肩膀旋轉角度（左肩-右肩連線與水平線夾角）
    返回: 旋轉角度 (度)
    """
    if keypoints is None or len(keypoints) < 17:
        return 0
    
    l_shoulder = keypoints[5]  # L_Shoulder
    r_shoulder = keypoints[6]  # R_Shoulder
    
    if l_shoulder[0] <= 0 or r_shoulder[0] <= 0:
        return 0
    
    dx = r_shoulder[0] - l_shoulder[0]
    dy = r_shoulder[1] - l_shoulder[1]
    
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    return angle


def calculate_body_side_angle(keypoints):
    """
    計算身體側身角度（基於兩肩與兩髖的深度差異）
    側身時，左右肩膀的 x 座標差距會變小（因為身體轉向側面）
    
    返回: 側身程度 (0-1)，1 表示完全側身，0 表示正面
    """
    if keypoints is None or len(keypoints) < 17:
        return 0
    
    l_shoulder = keypoints[5]  # L_Shoulder
    r_shoulder = keypoints[6]  # R_Shoulder
    l_hip = keypoints[11]      # L_Hip
    r_hip = keypoints[12]      # R_Hip
    
    # 檢查關鍵點有效性
    if (l_shoulder[0] <= 0 or r_shoulder[0] <= 0 or 
        l_hip[0] <= 0 or r_hip[0] <= 0):
        return 0
    
    # 計算肩膀寬度和髖部寬度
    shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
    hip_width = abs(r_hip[0] - l_hip[0])
    
    # 計算身體中心線（肩膀中點到髖部中點）
    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center = (l_hip + r_hip) / 2
    
    # 正常情況下，肩寬應該相對較大
    # 側身時，肩寬會變小（因為是側面視角）
    # 這裡用比例來判斷側身程度
    avg_width = (shoulder_width + hip_width) / 2
    
    # 如果平均寬度很小，表示側身
    # 正面時寬度通常 > 100 pixels，側身時 < 50 pixels
    if avg_width < 80:
        side_angle = 1.0 - (avg_width / 80.0)  # 0-1 之間
    else:
        side_angle = 0.0
    
    return min(1.0, max(0.0, side_angle))


def is_arm_raised(keypoints):
    """
    判斷手臂是否抬起（用於擊球準備動作）
    手腕明顯高於肩膀時，判定為抬臂
    
    返回: True/False
    """
    if keypoints is None or len(keypoints) < 17:
        return False
    
    r_shoulder = keypoints[6]  # R_Shoulder
    r_wrist = keypoints[10]    # R_Wrist
    
    if r_shoulder[0] <= 0 or r_wrist[0] <= 0:
        return False
    
    # 手腕 y 座標小於肩膀 y 座標（y軸向下為正）
    # 表示手腕在肩膀上方
    return r_wrist[1] < (r_shoulder[1] - 30)  # 至少高於肩膀 30 pixels


def is_jumping(keypoints, threshold=0.85):
    """
    判斷是否處於跳躍狀態
    返回: True/False
    """
    if keypoints is None or len(keypoints) < 17:
        return False
    
    hip = keypoints[12]    # R_Hip
    ankle = keypoints[16]  # R_Ankle
    
    if hip[0] <= 0 or ankle[0] <= 0:
        return False
    
    # 如果髖部明顯高於正常站立高度（相對於腳踝）
    # 正常站立時 hip_y > ankle_y，跳躍時差距縮小
    return hip[1] < ankle[1] * threshold


def classify_shot_type(keypoints, prev_keypoints):
    """
    根據姿勢特徵分類擊球類型
    
    改進版本：加入側身角度和抬臂判斷
    - 側身 + 抬臂 = 可能是擊球動作（殺球/高遠/切球）
    - 沒有側身或沒抬臂 = 可能只是移動
    
    Returns:
        str: 'smash' (殺球), 'clear' (高遠球), 'drop' (切球/放小球), 'unknown'
    """
    if keypoints is None or prev_keypoints is None:
        return 'unknown'
    
    if len(keypoints) < 17 or len(prev_keypoints) < 17:
        return 'unknown'
    
    # === 核心判斷：側身 + 抬臂 ===
    side_angle = calculate_body_side_angle(keypoints)
    arm_raised = is_arm_raised(keypoints)
    
    # 如果沒有側身且沒有抬臂，直接判定為非擊球動作
    if side_angle < 0.3 and not arm_raised:
        return 'unknown'
    
    # 1. 計算手腕速度（揮拍速度）
    wrist_speed = calculate_wrist_speed(keypoints, prev_keypoints)
    
    # 2. 手臂伸展角度（手肘角度）
    r_shoulder = keypoints[6]
    r_elbow = keypoints[8]
    r_wrist = keypoints[10]
    elbow_angle = angle_3points(r_shoulder, r_elbow, r_wrist)
    
    # 3. 身體後仰角度
    body_lean = calculate_body_lean(keypoints)
    
    # 4. 是否跳躍
    jumping = is_jumping(keypoints)
    
    # 5. 肩膀旋轉角度
    shoulder_rotation = calculate_shoulder_rotation(keypoints)
    
    # === 分類邏輯（加入側身考量）===
    
    # 殺球特徵：側身 + 抬臂 + 高速 + 跳躍 + 大幅後仰 + 手臂完全伸直
    if (side_angle > 0.4 and 
        arm_raised and
        wrist_speed > 25 and 
        jumping and 
        body_lean > 20 and 
        elbow_angle > 155):
        return 'smash'
    
    # 高遠球：側身 + 抬臂 + 中速 + 手臂伸直 + 明顯後仰 + 無跳躍
    elif (side_angle > 0.3 and
          arm_raised and
          wrist_speed > 15 and 
          wrist_speed < 35 and
          elbow_angle > 150 and 
          body_lean > 12 and 
          not jumping):
        return 'clear'
    
    # 切球/放小球：側身 + 抬臂 + 低速 + 手臂未完全伸直 + 小後仰
    elif (side_angle > 0.3 and
          arm_raised and
          wrist_speed < 22 and 
          elbow_angle < 155 and 
          body_lean < 18):
        return 'drop'
    
    else:
        return 'unknown'


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
    prev_keypoints = None
    wrist_speeds = []
    shot_detected = False
    shot_type = 'unknown'
    shot_cooldown = 0  # 防止重複偵測
    
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
        
        # === 2. 擊球瞬間偵測 ===
        if keypoints is not None and prev_keypoints is not None:
            wrist_speed = calculate_wrist_speed(keypoints, prev_keypoints)
            wrist_speeds.append(wrist_speed)
            
            # 偵測速度峰值（擊球瞬間）
            if shot_cooldown <= 0 and wrist_speed > 20:  # 速度閾值
                # 檢查是否為局部最大值
                if len(wrist_speeds) >= 3:
                    if wrist_speeds[-2] > wrist_speeds[-3] and wrist_speeds[-2] > wrist_speeds[-1]:
                        # 偵測到擊球瞬間
                        shot_detected = True
                        shot_type = classify_shot_type(prev_keypoints, keypoints)
                        shot_cooldown = 30  # 30幀內不再偵測（約1秒）
                        
                        print(f"\n🎯 Frame {frame_id}: 偵測到擊球! 類型: {shot_type.upper()}")
                        print(f"   手腕速度: {wrist_speeds[-2]:.1f} px/frame")
        
        # 冷卻計時器遞減
        if shot_cooldown > 0:
            shot_cooldown -= 1
            
            # 在冷卻期間顯示偵測結果
            if shot_cooldown > 20:  # 顯示10幀
                shot_color = {
                    'smash': (0, 0, 255),    # 紅色
                    'clear': (0, 255, 0),    # 綠色
                    'drop': (255, 0, 0),     # 藍色
                    'unknown': (128, 128, 128)
                }.get(shot_type, (255, 255, 255))
                
                cv2.putText(frame, f"SHOT: {shot_type.upper()}", 
                           (w//2 - 150, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, shot_color, 4)
        
        # === 3. 球場線條 ===
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
        
        # 更新前一幀關鍵點
        if keypoints is not None:
            prev_keypoints = keypoints.copy()
        
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
