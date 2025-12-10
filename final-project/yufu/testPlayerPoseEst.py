import cv2
import numpy as np
import torch
import torch.serialization
from ultralytics import YOLO
import time
from collections import deque
import json
from datetime import datetime
import sys
import argparse
from shot_trajectory_logger import TrajectoryLogger  # ⭐ 新增：軌跡記錄器

# Fix for PyTorch 2.6+ weights_only loading issue
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.PoseModel'])

# ===================== 參數區：請依你的環境修改 =====================
# 影片路徑（可被命令列參數覆蓋）
VIDEO_PATH = "./20250711_short.mp4"

# 輸出影片路徑（可被命令列參數覆蓋）
OUTPUT_PATH = "./output_with_pose.mp4"

# 球場標註座標 (np.save 出來的 court_pts.npy)
COURT_PTS_PATH = "./court_pts.npy"

# ⭐ 羽球邊界線座標（可選，如果有的話會用手動標註的，沒有就自動計算）
BALL_BOUNDARY_PATH = "./ball_boundary.npy"

# ⭐ 改成 YOLO Pose 權重（人 + skeleton）
YOLO_WEIGHTS = "./yolov8n-pose.pt"

# ⭐ 羽球偵測模型（新訓練的 YOLO 模型）
SHUTTLECOCK_WEIGHTS = "./runs/detect/shuttlecock_improved_20251209_122742/weights/best.pt"
USE_YOLO_SHUTTLECOCK = True  # 是否使用 YOLO 羽球偵測（可切換回差分偵測）
SHUTTLECOCK_CONF = 0.15  # ⭐ 降低置信度門檻（從 0.25 降到 0.15）讓更多候選被考慮
SHOW_ALL_DETECTIONS = True  # ⭐ 顯示所有偵測結果（包括低信心度的）

# 球偵測參數（請用你在滑桿小工具上調好的那組數值）
BALL_DIFF_THRESH = 40      # ⭐ 大幅降低門檻值，提高敏感度（從 70 降到 40）
BALL_BRIGHT_THRESH = 180   # ⭐ 降低亮度門檻，讓稍暗的球也能被偵測
BALL_MIN_AREA = 2          # ⭐ 進一步降低最小面積（從 3 降到 2）
BALL_MAX_AREA = 120        # ⭐ 限制最大面積（避免誤認球拍為球）
USE_BRIGHTNESS = False     # ⭐ False = 只看差分（比較接近你 slider 的直覺）

# ⭐ 新增：球的運動限制參數
MAX_BALL_SPEED = 2000      # ⭐ 放寬最大速度限制（從 1500 提高到 2000）
MAX_BALL_Y = 650           # ⭐ 擴大偵測範圍（從 600 提高到 650）
MIN_BALL_Y = 30            # ⭐ 降低最小 y 值（從 50 降到 30）
BALL_PREDICTION_WEIGHT = 0.3  # 卡爾曼預測權重（0-1，越大越信任預測）

# ⭐ 球與球拍分離參數
WRIST_EXCLUSION_RADIUS = 100  # 排除手腕周圍的半徑（像素）
RACKET_SHAPE_ASPECT_RATIO = 2.5  # 球拍長寬比門檻
# =============================================================

# COCO 17-keypoint skeleton 連線定義（適用 yolov8n-pose）
KPT_PAIRS = [
    (5, 7), (7, 9),      # 左手：肩-肘-腕
    (6, 8), (8, 10),     # 右手
    (11, 13), (13, 15),  # 左腳：髖-膝-踝
    (12, 14), (14, 16),  # 右腳
    (5, 6),              # 雙肩
    (11, 12),            # 雙髖
    (5, 11), (6, 12),    # 身體兩側
]


def find_ball_motion(frame, prev_gray=None, last_pos=None,
                     diff_thresh=25,      # 差分閾值：判斷像素變化的門檻
                     bright_thresh=200,   # 亮度閾值：判斷像素是否夠亮 (0-255)
                     min_area=5, max_area=400,  # 候選球的面積範圍（過濾雜訊）
                     use_brightness=True,  # 是否同時考慮亮度條件
                     predicted_pos=None,   # ⭐ 卡爾曼濾波預測的位置
                     max_speed=800,        # ⭐ 最大速度限制
                     min_y=100, max_y=400,  # ⭐ y 軸範圍限制
                     person_keypoints=None,  # ⭐ 人的關鍵點（用於排除手腕附近）
                     last_area=None):  # ⭐ 上一幀的面積（用於穩定性檢查）
    """
    利用前後幀差分 +（可選）亮度 + 面積偵測羽球
    
    原理：
    1. 前後幀差分：找出移動的物體
    2. 亮度篩選：羽球通常是白色或黃色，較亮
    3. 面積過濾：排除太大或太小的候選區域
    4. 距離追蹤：選擇離上一幀位置最近的候選
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        h, w = gray.shape
        debug_vis = np.zeros((h, w, 3), dtype=np.uint8)
        return None, None, gray, debug_vis, []

    # ---- 1) 前後幀差分 ----
    diff = cv2.absdiff(gray, prev_gray)
    _, diff_mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    # ---- 2) 亮度 mask ----
    _, bright_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)

    # ---- 3) 合併 (又亮又有動的點) ----
    moving_bright = cv2.bitwise_and(diff_mask, bright_mask)

    # ---- 4) 形態學去雜訊 ----
    kernel = np.ones((3, 3), np.uint8)
    diff_mask_filt = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    moving_bright_filt = cv2.morphologyEx(moving_bright, cv2.MORPH_OPEN, kernel, iterations=1)

    # ==== 決定用哪一個 mask 當連通區來源 ====
    if use_brightness:
        cc_src = moving_bright_filt      # 有動又夠亮
    else:
        cc_src = diff_mask_filt          # ⭐ 只看有動的

    # ---- 5) 連通區塊 ----
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cc_src)

    best_pos = None
    best_bbox = None
    min_dist = 1e9
    candidate_boxes = []

    for i in range(1, num_labels):  # 0 是背景
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # 1. 面積過濾
        if area < min_area or area > max_area:
            continue
        
        # 2. ⭐ 形狀過濾（排除長條形物體如球拍）
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        if aspect_ratio > 2.5:  # 球應該接近圓形，長寬比不應超過2.5
            continue
        
        # 3. ⭐ 圓形度檢查（使用面積和周長）
        # 提取該連通區域的輪廓
        region_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.3:  # 圓形度太低，可能是球拍
                    continue
        
        # 4. ⭐ 高度過濾（排除過高或過低的候選）
        if cy < min_y or cy > max_y:
            continue
        
        # 5. ⭐ 面積穩定性檢查（球的面積不應突然變化太大）
        if last_area is not None:
            area_ratio = area / (last_area + 1e-6)
            if area_ratio < 0.3 or area_ratio > 3.0:  # 面積變化超過3倍，可能是球拍
                continue
        
        # 6. ⭐ 速度過濾（避免異常跳躍）
        if last_pos is not None:
            speed = np.hypot(cx - last_pos[0], cy - last_pos[1])
            if speed > max_speed:
                continue  # 跳過速度異常的候選
        
        # 7. ⭐ 排除手腕附近區域（擊球瞬間球拍會在手腕附近）
        if person_keypoints is not None:
            too_close_to_wrist = False
            for kpts in person_keypoints:
                if len(kpts) >= 17:  # 確保有手腕關鍵點
                    # 檢查左右手腕和手肘（索引 7,8,9,10）
                    # 7=左肘, 8=右肘, 9=左腕, 10=右腕
                    for joint_idx in [7, 8, 9, 10]:
                        joint_x, joint_y = kpts[joint_idx]
                        if joint_x > 0 and joint_y > 0:
                            dist_to_joint = np.hypot(cx - joint_x, cy - joint_y)
                            # 使用全域參數
                            if dist_to_joint < 100:  # 距離手部關節100像素內，可能是球拍
                                too_close_to_wrist = True
                                break
                if too_close_to_wrist:
                    break
            if too_close_to_wrist:
                continue

        candidate_boxes.append((x, y, w, h, cx, cy, area))

        # 4. ⭐ 改進選擇邏輯：優先使用預測位置，其次用上一幀位置
        if last_pos is None and predicted_pos is None:
            best_pos = (cx, cy)
            best_bbox = (x, y, w, h)
        else:
            # 計算到參考點的距離（優先用預測位置）
            if predicted_pos is not None:
                ref_pos = predicted_pos
            else:
                ref_pos = last_pos
            
            dist = np.hypot(cx - ref_pos[0], cy - ref_pos[1])
            if dist < min_dist:
                min_dist = dist
                best_pos = (cx, cy)
                best_bbox = (x, y, w, h)

    # debug：R= moving_bright, G= diff_mask, B= bright_mask
    debug_vis = cv2.merge([
        moving_bright,
        diff_mask,
        bright_mask
    ])

    return best_pos, best_bbox, gray, debug_vis, candidate_boxes


def point_in_court(x, y, contour):
    val = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
    return val >= 0


def point_in_box(px, py, box):
    """檢查點 (px, py) 是否落在人的 bounding box 內"""
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def draw_skeleton(frame, kpts, color=(0, 255, 255)):
    """
    在 frame 上畫出一個人的 skeleton
    kpts: shape (K, 2)，COCO 17 keypoints
    """
    # 關節點
    for x, y in kpts:
        if x <= 0 or y <= 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    # 連線
    for i, j in KPT_PAIRS:
        if i >= len(kpts) or j >= len(kpts):
            continue
        x1, y1 = kpts[i]
        x2, y2 = kpts[j]
        if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
            continue
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


# ===================== AI 輔助判斷系統 =====================
# 記錄檔案路徑
SHOT_LOG_FILE = "./shot_annotations.json"

def load_shot_history():
    """載入歷史擊球記錄"""
    try:
        with open(SHOT_LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_shot_record(frame_idx, params, predicted_type, user_label=None):
    """儲存擊球記錄"""
    history = load_shot_history()
    
    # 將 numpy 類型轉換為 Python 原生類型
    params_serializable = {}
    for key, value in params.items():
        if isinstance(value, (np.integer, np.floating)):
            params_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            params_serializable[key] = value.tolist()
        else:
            params_serializable[key] = value
    
    record = {
        'timestamp': datetime.now().isoformat(),
        'frame': int(frame_idx),
        'parameters': params_serializable,
        'predicted': predicted_type,
        'user_label': user_label,
        'correct': user_label == predicted_type if user_label else None
    }
    history.append(record)
    with open(SHOT_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    return record

def get_ai_suggestion(params):
    """
    基於歷史數據的 AI 建議（簡單版本）
    未來可以升級為機器學習模型
    """
    history = load_shot_history()
    if len(history) < 5:  # 數據不足，使用規則
        return None, 0.0
    
    # 找到相似的歷史案例（有用戶標註的）
    labeled_history = [h for h in history if h.get('user_label')]
    if not labeled_history:
        return None, 0.0
    
    # 簡單相似度匹配（可以改進為更複雜的 ML 模型）
    similarities = []
    for record in labeled_history:
        hist_params = record['parameters']
        # 計算參數相似度
        slope_diff = abs(params['overall_slope'] - hist_params['overall_slope'])
        vel_diff = abs(params['velocity'] - hist_params['velocity'])
        pos_diff = abs(params['highest_position_ratio'] - hist_params['highest_position_ratio'])
        
        # 簡單加權相似度
        similarity = 1.0 / (1.0 + slope_diff/50 + vel_diff/200 + pos_diff)
        similarities.append((similarity, record['user_label']))
    
    # 找最相似的案例
    if similarities:
        similarities.sort(reverse=True)
        best_match = similarities[0]
        return best_match[1], best_match[0]  # (建議類型, 信心度)
    
    return None, 0.0

# =============================================================


def is_arm_raised(kpts):
    """
    判斷右手臂是否抬起（用於偵測擊球動作的起始時刻）
    
    判斷邏輯：
    1. 右手腕必須在右肩上方至少 40 pixels
    2. 右肘不能低於右肩太多（允許 20 pixels 容錯）
    
    COCO 17 keypoints 索引：
    - 6 = 右肩
    - 8 = 右肘
    - 10 = 右手腕
    
    Returns: 
        True - 右手臂抬起（偵測到擊球準備動作）
        False - 右手臂未抬起
    """
    if len(kpts) < 17:
        return False
    
    r_shoulder = kpts[6]  # 右肩
    r_elbow = kpts[8]     # 右肘
    r_wrist = kpts[10]    # 右手腕
    
    # 檢查關鍵點是否有效（信心度>0）
    if r_shoulder[0] <= 0 or r_shoulder[1] <= 0:
        return False
    if r_wrist[0] <= 0 or r_wrist[1] <= 0:
        return False
    
    # 右手腕 y 座標小於右肩（y 軸向下為正）表示右手腕在右肩上方
    # 增加門檻值到 50 pixels，確保確實是抬手動作
    wrist_above_shoulder = r_wrist[1] < (r_shoulder[1] - 40)
    
    # 額外檢查：右肘也應該在右肩上方（更嚴格的抬手判定）
    elbow_check = True
    if r_elbow[0] > 0 and r_elbow[1] > 0:
        elbow_check = r_elbow[1] < (r_shoulder[1] + 20)  # 右肘至少不低於右肩太多
    
    return wrist_above_shoulder and elbow_check


def classify_shot_by_trajectory(ball_history, shot_frame_idx, fps=30, 
                                  frames_after=5, frames_before=3):
    """
    根據球的軌跡判斷擊球類型
    
    Args:
        ball_history: deque of (frame_idx, cx, cy) or None
        shot_frame_idx: 擊球發生的幀數
        fps: 影片 fps
        frames_after: 擊球後取幾幀來計算（現在使用 45 幀）
        frames_before: 擊球前取幾幀來確認
    
    Returns:
        shot_type: "Smash", "Clear", "Drop", or "Unknown"
        trajectory_info: dict with Δx, Δy, velocity, angle
    """
    # 從 ball_history 中找到擊球前後的球位置
    ball_at_shot = None
    ball_after = None
    
    # 找擊球時刻的球位置（允許前後 2 幀的容錯）
    for entry in ball_history:
        if entry is None:
            continue
        frame_idx, cx, cy = entry
        if abs(frame_idx - shot_frame_idx) <= 2:
            ball_at_shot = (cx, cy)
            break
    
    # 找擊球後的球位置（取第 frames_after 幀）
    target_frame = shot_frame_idx + frames_after
    for entry in ball_history:
        if entry is None:
            continue
        frame_idx, cx, cy = entry
        if abs(frame_idx - target_frame) <= 2:
            ball_after = (cx, cy)
            break
    
    if ball_at_shot is None or ball_after is None:
        return "Unknown", {}
    
    # 計算 Δx, Δy
    x0, y0 = ball_at_shot
    x1, y1 = ball_after
    dx = x1 - x0
    dy = y1 - y0
    
    # 計算速度（pixels per second）
    dt = frames_after / fps  # 時間間隔（秒）
    distance = np.hypot(dx, dy)
    velocity = distance / dt if dt > 0 else 0
    
    # 計算角度（相對於水平線，向下為正）
    angle_deg = np.degrees(np.arctan2(dy, abs(dx))) if dx != 0 else 0
    
    # === 分類邏輯 ===
    # 門檻值（可調整）
    SMASH_VELOCITY_THRESH = 800    # 殺球速度門檻（pixels/s）
    CLEAR_DY_THRESH = -50          # 高遠球：Δy 明顯為負（往上飛）
    DROP_VELOCITY_THRESH = 400     # 切球速度門檻
    
    shot_type = "Unknown"
    
    if dy < CLEAR_DY_THRESH:
        # 球往上飛 → 高遠球
        shot_type = "Clear"
    elif dy > 0 and velocity > SMASH_VELOCITY_THRESH:
        # 球往下且速度很快 → 殺球
        shot_type = "Smash"
    elif dy > 0 and velocity < DROP_VELOCITY_THRESH:
        # 球往下但速度慢 → 切球
        shot_type = "Drop"
    elif velocity > SMASH_VELOCITY_THRESH:
        # 速度很快但方向不明顯 → 可能是平抽或殺球
        shot_type = "Smash/Drive"
    else:
        # 其他情況
        shot_type = "Drop/Net"
    
    trajectory_info = {
        'dx': dx,
        'dy': dy,
        'velocity': velocity,
        'angle': angle_deg,
        'ball_at_shot': ball_at_shot,
        'ball_after': ball_after
    }
    
    return shot_type, trajectory_info


def main():
    global VIDEO_PATH, OUTPUT_PATH
    
    # === 命令列參數解析 ===
    parser = argparse.ArgumentParser(
        description='羽球影片分析系統 - 標註訓練版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例:
  python testPlayerPoseEst.py                              # 使用預設影片
  python testPlayerPoseEst.py -i input.mp4                 # 指定輸入影片
  python testPlayerPoseEst.py -i input.mp4 -o output.mp4   # 指定輸入和輸出
        ''')
    parser.add_argument('-i', '--input', type=str, default=VIDEO_PATH,
                       help=f'輸入影片路徑 (預設: {VIDEO_PATH})')
    parser.add_argument('-o', '--output', type=str, default=OUTPUT_PATH,
                       help=f'輸出影片路徑 (預設: {OUTPUT_PATH})')
    args = parser.parse_args()
    
    # 使用命令列參數或預設值
    VIDEO_PATH = args.input
    OUTPUT_PATH = args.output
    
    print(f"📹 輸入影片: {VIDEO_PATH}")
    print(f"📤 輸出影片: {OUTPUT_PATH}")
    
    # === 設定 GPU 加速 (MPS for M4 Pro) ===
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ 使用 MPS (Metal) GPU 加速")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("✅ 使用 CUDA GPU 加速")
    else:
        device = 'cpu'
        print("⚠️  使用 CPU")
    
    # === 讀取球場 polygon ===
    court_pts = np.load(COURT_PTS_PATH)
    court_contour = court_pts.reshape((-1, 1, 2)).astype(np.int32)

    # === 讀取或計算球的左右邊界 ===
    try:
        # 嘗試讀取手動標註的邊界線
        boundary_data = np.load(BALL_BOUNDARY_PATH, allow_pickle=True).item()
        x_min_ball = boundary_data['left_x']
        x_max_ball = boundary_data['right_x']
        print(f"✅ 使用手動標註的邊界線: left_x={x_min_ball}, right_x={x_max_ball}")
    except:
        # 沒有手動標註的話，自動計算（用球場最上面兩點）
        print("⚠️  沒有找到手動標註的邊界線，使用自動計算...")
        ys = court_pts[:, 1]
        idx_sorted = np.argsort(ys)        # y 由小到大
        top_idx = idx_sorted[:2]           # 最上面兩點（y 最小）
        top_pts = court_pts[top_idx]       # shape (2, 2)
        print("Top points (far side) for ball:", top_pts)

        x_min = top_pts[:, 0].min()
        x_max = top_pts[:, 0].max()

        margin = 5  # 預留一點空間
        x_min_ball = int(x_min + margin)
        x_max_ball = int(x_max - margin)

        print("Court x range for ball (using top points):", x_min_ball, "to", x_max_ball)

    # === 影片 IO ===
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, size: {w}x{h}, total frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 使用較低的品質以加快編碼速度
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
    if not out.isOpened():
        print("⚠️  警告：無法創建輸出影片")
    print("Output will be saved to:", OUTPUT_PATH)

    # === YOLO Pose 抓人 + skeleton ===
    print(f"載入人物姿態模型: {YOLO_WEIGHTS}")
    people_model = YOLO(YOLO_WEIGHTS)
    people_model.to(device)  # 將模型移到 GPU
    
    # === ⭐ YOLO 羽球偵測模型 ===
    shuttlecock_model = None
    use_yolo_shuttlecock = USE_YOLO_SHUTTLECOCK  # 建立局部變數
    if use_yolo_shuttlecock:
        try:
            print(f"載入羽球偵測模型: {SHUTTLECOCK_WEIGHTS}")
            shuttlecock_model = YOLO(SHUTTLECOCK_WEIGHTS)
            shuttlecock_model.to(device)
            print(f"✅ 羽球偵測模型已載入（置信度門檻: {SHUTTLECOCK_CONF}）")
        except Exception as e:
            print(f"⚠️  無法載入羽球偵測模型: {e}")
            print("將使用傳統差分偵測方法")
            use_yolo_shuttlecock = False
    
    # 啟用 FP16 半精度運算（MPS 支援，可大幅加速）
    if device == 'mps':
        print("啟用 FP16 半精度運算以加速推論")
    
    print(f"模型已載入到: {device}")

    prev_gray = None
    last_ball = None
    ball_bbox = None  # ⭐ 初始化球的邊界框
    frame_idx = 0

    # === 球軌跡記錄（用於判斷擊球類型）===
    ball_history = deque(maxlen=90)  # 保留最近 90 幀的球位置（3秒@30fps，確保 45 幀判斷有足夠歷史）
    
    # ⭐ 新的軌跡記錄器（記錄完整 x, y 座標）
    trajectory_logger = TrajectoryLogger("./shot_trajectories.json")
    
    # ⭐ 卡爾曼濾波器（用於球位置預測）
    ball_velocity = None  # 球的速度向量 (vx, vy)
    predicted_ball_pos = None  # 預測的球位置

    # === 手臂動作追蹤變數 ===
    arm_was_raised = False      # 上一幀手臂是否抬起
    arm_raised_frame = None     # 手臂抬起的幀數（用於追蹤後續球的運動）
    arm_cooldown_until = 0      # ⭐ 冷卻時間：在此幀數之前不接受新的 arm up（避免重複觸發）
    shot_detected = False       # 是否偵測到擊球動作
    shot_type = ""             # 擊球類型
    shot_display_timer = 0      # 顯示計時器（顯示 30 幀 = 約 1 秒）
    trajectory_info = {}        # 軌跡資訊
    tracking_points = []        # 追蹤期間的所有球位置點（用於可視化）
    analysis_complete = False   # 判斷是否已完成分析

    # === 進度追蹤變數 ===
    start_time = time.time()
    last_progress_time = start_time
    print("\n開始處理影片...")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 先畫球場 polygon（簡化線條粗細）
        cv2.polylines(frame, [court_contour], isClosed=True, color=(255, 0, 0), thickness=1)

        # 畫出「球的 x 範圍」兩條線（遠端左右線，更細的線）
        cv2.line(frame, (x_min_ball, 0), (x_min_ball, int(h - 1)), (0, 255, 255), 1)
        cv2.line(frame, (x_max_ball, 0), (x_max_ball, int(h - 1)), (0, 255, 255), 1)

        # --- 1) YOLO Pose 抓「人」 + skeleton ---
        person_boxes = []  # 存所有人的 bbox（不管在不在場內，用來過濾球）
        text_boxes = []    # 存所有文字標註的 bbox（用來過濾球的候選區域）
        predicted_pos_from_slope = None  # ⭐ 初始化軌跡預測位置（供選球策略使用）

        # 使用 GPU 加速推論（保持 640 解析度以維持品質）
        results = people_model(
            frame, 
            conf=0.5, 
            imgsz=640,  # 保持 640 以維持偵測品質
            verbose=False, 
            device=device,
            half=True if device == 'mps' else False,
            max_det=10,
            agnostic_nms=True
        )
        result = results[0]

        has_kpts = (getattr(result, "keypoints", None) is not None)
        if has_kpts:
            # 直接在 GPU 上操作，減少 CPU-GPU 資料傳輸
            kpts_all = result.keypoints.xy  # 保持在 GPU 上
        else:
            kpts_all = None

        # 先收集所有人的資訊
        all_persons = []  # 存所有人的資訊 (index, bbox, conf, in_court)
        
        # 一次性將所有 keypoints 轉到 CPU（減少重複轉換）
        if has_kpts and len(kpts_all) > 0:
            kpts_all_cpu = kpts_all.cpu().numpy()
        else:
            kpts_all_cpu = None
        
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != 0:  # 只看 person 類別
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

            # 不論在不在場內，都要記錄，之後拿來排除球的 candidate
            person_boxes.append((x1i, y1i, x2i, y2i))

            # 用骨架的腳踝關鍵點來判斷是否在場內（更準確）
            inside_person = False
            bl = (x1i, y2i)  # 預設用 bbox 底部
            br = (x2i, y2i)
            
            if kpts_all_cpu is not None and i < len(kpts_all_cpu):
                kpts = kpts_all_cpu[i]  # 使用預先轉換的 CPU 版本
                # COCO keypoints: 15=左腳踝, 16=右腳踝
                left_ankle = kpts[15]   # (x, y)
                right_ankle = kpts[16]  # (x, y)
                
                # 更新視覺化位置（使用實際腳踝）
                if left_ankle[0] > 0 and left_ankle[1] > 0:
                    bl = (int(left_ankle[0]), int(left_ankle[1]))
                    if point_in_court(left_ankle[0], left_ankle[1], court_contour):
                        inside_person = True
                        
                if right_ankle[0] > 0 and right_ankle[1] > 0:
                    br = (int(right_ankle[0]), int(right_ankle[1]))
                    if point_in_court(right_ankle[0], right_ankle[1], court_contour):
                        inside_person = True
            else:
                # 如果沒有骨架資訊，fallback 到 bbox 底部兩點
                in_bl = point_in_court(*bl, court_contour)
                in_br = point_in_court(*br, court_contour)
                inside_person = in_bl or in_br  # 只要任一點在場內就算

            # 計算 bbox 面積
            bbox_area = (x2i - x1i) * (y2i - y1i)
            
            all_persons.append({
                'index': i,
                'bbox': (x1i, y1i, x2i, y2i),
                'conf': conf,
                'in_court': inside_person,
                'area': bbox_area,
                'bl': bl,
                'br': br
            })

        # 決定要畫哪一個人的優先順序：
        # 1. 最優先：腳踝在場內的人（如果有多個，選面積最大的）
        # 2. 次要：如果沒有人的腳踝在場內，才選面積最大的
        selected_person = None
        persons_in_court = [p for p in all_persons if p['in_court']]
        
        if persons_in_court:
            # 優先：有人在場內，選其中面積最大的
            selected_person = max(persons_in_court, key=lambda p: p['area'])
        elif all_persons:
            # 次要：沒有人在場內，選所有人中面積最大的
            selected_person = max(all_persons, key=lambda p: p['area'])
        
        # === 手臂動作偵測（只對選中的人做） ===
        current_arm_raised = False
        if selected_person and kpts_all_cpu is not None:
            i = selected_person['index']
            kpts = kpts_all_cpu[i]  # 使用預先轉換的版本
            current_arm_raised = is_arm_raised(kpts)
            
            # ⭐ 冷卻機制：檢查是否在冷卻期間內
            # 記錄手臂抬起的時刻（只有右手臂抬起才觸發，且不在冷卻期）
            if current_arm_raised and not arm_was_raised and frame_idx >= arm_cooldown_until:
                # 右手臂從放下變成抬起，且已過冷卻期
                arm_raised_frame = frame_idx
                arm_cooldown_until = frame_idx + 10  # ⭐ 設定冷卻時間：接下來10幀內不接受新的 arm up
                tracking_points = []  # 清空追蹤點
                analysis_complete = False  # 重置分析狀態
                shot_display_timer = 0  # 清空之前的顯示計時器
                trajectory_info = {}  # 清空之前的軌跡資訊
                print(f"\n📍 Frame {frame_idx}: 右手臂抬起，開始追蹤球的運動...")
                print(f"   ⏱️  冷卻時間：Frame {frame_idx} ~ {arm_cooldown_until} (10幀) 內不接受新的手臂抬起")
                print(f"   📊 追蹤計畫：等待 5 幀 + 追蹤 45 幀 = 總共 50 幀")
            
            # ⚠️ 注意：tracking_points 的收集移到偵測球之後（在後面的代碼中）
            
            # ⭐ 在右手臂抬起後等待 55 幀（5幀等待 + 50幀追蹤）檢查球的運動方向
            # 前5幀讓球飛出去，不計入判定；後50幀用於分析軌跡
            if arm_raised_frame is not None and not analysis_complete and (frame_idx - arm_raised_frame) >= 55:
                # 分析從手臂抬起到現在的球運動方向
                ball_at_raise = None
                ball_now = None
                
                # 找手臂抬起時刻附近的球位置（容錯±3幀）
                for entry in ball_history:
                    if entry is None:
                        continue
                    f_idx, cx, cy = entry
                    if abs(f_idx - arm_raised_frame) <= 3:
                        ball_at_raise = (cx, cy)
                        break
                
                # 找現在的球位置（最近的球）
                for entry in reversed(ball_history):
                    if entry is not None:
                        f_idx, cx, cy = entry
                        ball_now = (cx, cy)
                        break
                
                if ball_at_raise and ball_now:
                    # 檢查是否有足夠的追蹤點
                    if len(tracking_points) == 0:
                        print(f"   ⚠️  警告：沒有追蹤到任何球的位置，跳過分析")
                        # 重置追蹤狀態
                        arm_raised_frame = None
                        tracking_points = []
                        analysis_complete = False
                        shot_display_timer = 0
                        trajectory_info = {}
                        continue
                    
                    # 計算 y 軸方向變化
                    dy = ball_now[1] - ball_at_raise[1]
                    dx = ball_now[0] - ball_at_raise[0]
                    
                    # 計算速度
                    dt = (frame_idx - arm_raised_frame) / fps
                    distance = np.hypot(dx, dy)
                    velocity = distance / dt if dt > 0 else 0
                    
                    # === 統計45幀追蹤期間有多少球在高處（y < 350)===
                    # 目的：判斷是否為高遠球（Clear）
                    # 高遠球特徵：球大部分時間停留在畫面高處
                    high_ball_count = 0  # 計數：在高處（y < 350）的球數量
                    total_tracked = len(tracking_points)
                    
                    for (px, py, f_idx) in tracking_points:
                        if py < 350:  # y 座標越小表示越靠近畫面頂部（高處）
                            high_ball_count += 1
                    
                    high_ball_ratio = high_ball_count / total_tracked if total_tracked > 0 else 0
                    
                    # === 檢查最後5幀是否在低處（y > 400）===
                    # 目的：排除「先上升後快速下降」的球（不是真正的高遠球）
                    # 真正的高遠球應該持續在高處，不會在追蹤末期出現在低處
                    last_frames_low = False
                    if len(tracking_points) >= 5:
                        last_5_frames = tracking_points[-5:]  # 取最後5幀
                        low_count = sum(1 for (px, py, f_idx) in last_5_frames if py > 400)
                        if low_count >= 3:  # 最後5幀中有3幀或以上在低處（y > 400）
                            last_frames_low = True  # 標記為「結束時在低處」
                    
                    # === 計算頭尾 y 軸位移（用於顯示）===
                    head_to_tail_dy = 0
                    if len(tracking_points) >= 2:
                        first_y = tracking_points[0][1]   # 起始位置的 y 座標
                        last_y = tracking_points[-1][1]   # 結束位置的 y 座標
                        head_to_tail_dy = last_y - first_y  # 正值 = 往下降，負值 = 往上升
                    
                    # === ⭐ 檢測球的轉折點（對方回擊）===
                    # 檢測軌跡中是否有明顯的方向轉折（可能是對方接到球並回擊）
                    has_turning_point = False
                    turning_point_info = ""
                    
                    if len(tracking_points) >= 10:  # 至少需要10個點才能分析轉折
                        # 計算每段的 y 方向變化
                        y_changes = []
                        for i in range(1, len(tracking_points)):
                            dy_segment = tracking_points[i][1] - tracking_points[i-1][1]
                            y_changes.append(dy_segment)
                        
                        # 檢測方向轉折：從下降變上升，或從上升變下降
                        # 使用滑動窗口（5個點）來平滑化並檢測趨勢變化
                        window_size = 5
                        if len(y_changes) >= window_size * 2:
                            for i in range(window_size, len(y_changes) - window_size):
                                # 計算前後窗口的平均斜率
                                before_slope = sum(y_changes[i-window_size:i]) / window_size
                                after_slope = sum(y_changes[i:i+window_size]) / window_size
                                
                                # 檢測明顯的方向轉折（門檻值可調整）
                                # 下降轉上升：before_slope > 10 且 after_slope < -10
                                # 上升轉下降：before_slope < -10 且 after_slope > 10
                                if (before_slope > 15 and after_slope < -15) or \
                                   (before_slope < -15 and after_slope > 15):
                                    has_turning_point = True
                                    turning_point_frame = tracking_points[i][2]
                                    turning_point_pos = i / len(tracking_points)
                                    if before_slope > 15 and after_slope < -15:
                                        turning_point_info = f"下降→上升 (位置:{turning_point_pos:.1%}, Frame:{turning_point_frame})"
                                    else:
                                        turning_point_info = f"上升→下降 (位置:{turning_point_pos:.1%}, Frame:{turning_point_frame})"
                                    break  # 找到第一個轉折點即可
                    
                    # === ⭐ 對於 Smash/Drop：只使用下降階段的軌跡（排除轉折後的上升） ===
                    # 找到最低點（y 最大的點），只用從開始到最低點的軌跡
                    if last_frames_low:  # 只有在判定為 Smash/Drop 時才需要過濾
                        lowest_idx = max(range(len(tracking_points)), key=lambda i: tracking_points[i][1])
                        # 只保留從起點到最低點的軌跡
                        descent_points = tracking_points[:lowest_idx + 1]
                    else:
                        # Clear 的話使用完整軌跡
                        descent_points = tracking_points
                    
                    # === ⭐ 使用軌跡判斷擊球類型 ===
                    
                    # 1. 使用過濾後的軌跡計算斜率
                    overall_slope = 0
                    if len(descent_points) >= 2:
                        first_y = descent_points[0][1]
                        last_y = descent_points[-1][1]
                        overall_slope = last_y - first_y  # 正值 = 往下降，負值 = 往上升
                    else:
                        # 如果過濾後軌跡不足，使用完整軌跡
                        overall_slope = head_to_tail_dy
                    
                    # 2. 計算最高點位置（使用完整軌跡）
                    min_y = min(pt[1] for pt in tracking_points)  # 最高點（y最小）
                    max_y = max(pt[1] for pt in tracking_points)  # 最低點（y最大）
                    y_range = max_y - min_y  # 垂直移動範圍
                    
                    # 3. 找到最高點的位置索引
                    highest_idx = min(range(len(tracking_points)), key=lambda i: tracking_points[i][1])
                    highest_position_ratio = highest_idx / len(tracking_points)  # 最高點出現的相對位置（0-1）
                    
                    # ⭐ 3.5 計算加速度（用於區分 Smash 和 Drop）
                    # Smash 通常有明顯加速，Drop 則速度較平穩
                    acceleration = 0
                    if len(tracking_points) >= 20:
                        # 比較前半和後半的平均速度
                        mid_idx = len(tracking_points) // 2
                        # 前半速度
                        first_half_dist = 0
                        for i in range(1, mid_idx):
                            dx = tracking_points[i][0] - tracking_points[i-1][0]
                            dy = tracking_points[i][1] - tracking_points[i-1][1]
                            first_half_dist += np.hypot(dx, dy)
                        first_half_speed = first_half_dist / mid_idx if mid_idx > 0 else 0
                        
                        # 後半速度
                        second_half_dist = 0
                        for i in range(mid_idx, len(tracking_points)):
                            dx = tracking_points[i][0] - tracking_points[i-1][0]
                            dy = tracking_points[i][1] - tracking_points[i-1][1]
                            second_half_dist += np.hypot(dx, dy)
                        second_half_frames = len(tracking_points) - mid_idx
                        second_half_speed = second_half_dist / second_half_frames if second_half_frames > 0 else 0
                        
                        # 加速度 = 後半速度 - 前半速度（正值表示加速）
                        acceleration = second_half_speed - first_half_speed
                    
                    # 4. ⭐ 綜合判斷擊球類型
                    # - Clear（高遠球）：整體向上或前期向上
                    # - Smash（殺球）：整體快速向下 + 有明顯加速
                    # - Drop（切球）：整體緩慢向下 + 無明顯加速
                    
                    # === 決策樹開始 ===
                    print("\n" + "="*60)
                    print(f"🎯 Frame {frame_idx}: 擊球分類決策樹")
                    print("="*60)
                    print(f"📊 分析參數:")
                    print(f"   - overall_slope (頭尾y變化): {overall_slope:.2f}")
                    if last_frames_low and len(descent_points) < len(tracking_points):
                        print(f"     ⚠️  使用下降階段軌跡 (排除轉折後上升): {len(descent_points)}/{len(tracking_points)} 幀")
                    print(f"   - highest_position_ratio (最高點位置): {highest_position_ratio:.2f}")
                    print(f"   - velocity (速度): {velocity:.2f} px/s")
                    print(f"   - acceleration (加速度): {acceleration:.2f} px/frame")
                    print(f"   - y_range (垂直範圍): {y_range:.2f}")
                    print(f"   - high_ball_ratio (高處停留比例): {high_ball_ratio:.2f}")
                    print(f"   - last_frames_low (最後5幀在低處): {last_frames_low}")
                    print(f"   - has_turning_point (軌跡有轉折): {has_turning_point}")
                    if has_turning_point:
                        print(f"     🔄 轉折資訊: {turning_point_info}")
                        print(f"     ⚠️  可能是對方回擊的球！")
                    print(f"\n🌲 決策過程:")
                    
                    # 先檢查最後5幀是否在低處 → 如果是，排除 Clear
                    if last_frames_low:
                        print(f"   ✓ last_frames_low = True (最後5幀有3幀以上在y>400)")
                        print(f"   → 排除 Clear（高遠球不應該結束在低處）")
                        if overall_slope > 80:  # ⭐ 提高門檻到80（更明顯的下降才判斷為攻擊球）
                            print(f"   ✓ overall_slope ({overall_slope:.2f}) > 80")
                            # ⭐ 使用加速度區分 Smash 和 Drop
                            if acceleration > 2 or velocity > 550:  # 有加速或高速度 → 殺球
                                shot_type = "Smash"
                                print(f"      ✓ acceleration ({acceleration:.2f}) > 2 或 velocity ({velocity:.2f}) > 550")
                                print(f"      → 有加速或高速度 → Smash (殺球)")
                            else:
                                shot_type = "Drop"  # 無加速且低速度 → 切球
                                print(f"      ✗ acceleration ({acceleration:.2f}) <= 2 且 velocity ({velocity:.2f}) <= 550")
                                print(f"      → 無加速且速度較低 → Drop (切球)")
                        else:
                            print(f"   ✗ overall_slope ({overall_slope:.2f}) <= 80")
                            # 下降不明顯，以 Drop 為主
                            shot_type = "Drop"
                            print(f"      → 下降不明顯 → Drop (切球)")
                    elif overall_slope < -30:  # 整體向上超過30像素 → 高遠球
                        shot_type = "Clear"
                        print(f"   ✗ last_frames_low = False")
                        print(f"   ✓ overall_slope ({overall_slope:.2f}) < -30")
                        print(f"   → 整體向上超過30像素 → Clear (高遠球)")
                    elif highest_position_ratio < 0.3:  # 最高點在前30% → 高遠球（球先上升）
                        shot_type = "Clear"
                        print(f"   ✗ last_frames_low = False")
                        print(f"   ✗ overall_slope ({overall_slope:.2f}) >= -30")
                        print(f"   ✓ highest_position_ratio ({highest_position_ratio:.2f}) < 0.3")
                        print(f"   → 最高點在前30% → Clear (高遠球)")
                    elif overall_slope > 80:  # ⭐ 提高門檻
                        print(f"   ✗ last_frames_low = False")
                        print(f"   ✗ overall_slope ({overall_slope:.2f}) >= -30")
                        print(f"   ✗ highest_position_ratio ({highest_position_ratio:.2f}) >= 0.3")
                        print(f"   ✓ overall_slope ({overall_slope:.2f}) > 80")
                        # ⭐ 使用加速度區分
                        if acceleration > 2 or velocity > 550:
                            shot_type = "Smash"
                            print(f"      ✓ acceleration ({acceleration:.2f}) > 2 或 velocity ({velocity:.2f}) > 550")
                            print(f"      → Smash (殺球)")
                        else:
                            shot_type = "Drop"
                            print(f"      ✗ acceleration ({acceleration:.2f}) <= 2 且 velocity ({velocity:.2f}) <= 550")
                            print(f"      → Drop (切球)")
                    else:
                        # 整體變化不大或輕微下降 → 根據速度判斷
                        print(f"   ✗ last_frames_low = False")
                        print(f"   ✗ overall_slope ({overall_slope:.2f}) >= -30")
                        print(f"   ✗ highest_position_ratio ({highest_position_ratio:.2f}) >= 0.3")
                        print(f"   ✗ overall_slope ({overall_slope:.2f}) <= 80")
                        print(f"   → 整體變化不大或輕微下降，根據速度判斷:")
                        if velocity > 600:
                            shot_type = "Smash"
                            print(f"      ✓ velocity ({velocity:.2f}) > 600")
                            print(f"      → Smash (殺球)")
                        else:
                            shot_type = "Clear"
                            print(f"      ✗ velocity ({velocity:.2f}) <= 600")
                            print(f"      → Clear (高遠球)")
                    
                    # === AI 輔助建議 ===
                    params_dict = {
                        'overall_slope': overall_slope,
                        'highest_position_ratio': highest_position_ratio,
                        'velocity': velocity,
                        'acceleration': acceleration,
                        'y_range': y_range,
                        'high_ball_ratio': high_ball_ratio,
                        'last_frames_low': last_frames_low,
                        'has_turning_point': has_turning_point,
                        'turning_point_info': turning_point_info if has_turning_point else ""
                    }
                    
                    ai_suggestion, confidence = get_ai_suggestion(params_dict)
                    
                    print(f"\n🤖 AI 建議:")
                    if ai_suggestion and confidence > 0.5:
                        print(f"   根據歷史數據建議: {ai_suggestion} (信心度: {confidence:.2f})")
                        if ai_suggestion != shot_type:
                            print(f"   ⚠️  AI建議與規則判斷不同！規則={shot_type}, AI={ai_suggestion}")
                    else:
                        print(f"   歷史數據不足，使用規則判斷")
                    
                    print(f"\n🏸 最終結果: {shot_type}")
                    print("="*60)
                    print(f"\n💡 提示: 按鍵標註正確答案 (標註後按 Space 繼續)")
                    print(f"   [C] = Clear  [S] = Smash  [D] = Drop  [Space] = 接受當前判斷")
                    print("-"*60 + "\n")
                    
                    # 暫停並等待用戶標註
                    cv2.putText(frame, "Press: [C]Clear [S]Smash [D]Drop [Space]Accept", 
                               (50, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, f"AI Predicted: {shot_type}", 
                               (50, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    cv2.imshow("Badminton Analysis", frame)
                    
                    user_label = None
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('c') or key == ord('C'):
                            user_label = "Clear"
                            print(f"✅ 用戶標註: Clear")
                            break
                        elif key == ord('s') or key == ord('S'):
                            user_label = "Smash"
                            print(f"✅ 用戶標註: Smash")
                            break
                        elif key == ord('d') or key == ord('D'):
                            user_label = "Drop"
                            print(f"✅ 用戶標註: Drop")
                            break
                        elif key == ord(' '):  # Space = 接受AI判斷
                            user_label = shot_type
                            print(f"✅ 接受AI判斷: {shot_type}")
                            break
                        elif key == ord('q') or key == ord('Q'):
                            print("⏭️  跳過標註")
                            break
                    
                    # 儲存記錄
                    if user_label:
                        save_shot_record(frame_idx, params_dict, shot_type, user_label)
                        # ⭐ 新增：儲存完整軌跡
                        trajectory_logger.save_shot(
                            end_frame=frame_idx,
                            user_label=user_label,
                            court_area=None,  # 如果有球場區域判斷，可以加入
                            metadata={
                                'predicted_type': shot_type,
                                'correct': user_label == shot_type,
                                'old_params': params_dict  # 保留舊格式特徵供參考
                            }
                        )
                        if user_label == shot_type:
                            print(f"✓ 判斷正確！")
                        else:
                            print(f"✗ 判斷錯誤！正確答案是 {user_label}，系統判斷為 {shot_type}")
                    
                    print("\n" + "="*60 + "\n")
                    
                    shot_detected = True
                    shot_display_timer = 75  # 顯示 75 幀（2.5秒），配合 45 幀追蹤時間
                    analysis_complete = True  # 標記分析完成
                    
                    trajectory_info = {
                        'dx': dx,
                        'dy': dy,
                        'velocity': velocity,
                        'ball_at_raise': ball_at_raise,
                        'ball_now': ball_now,
                        'high_ball_ratio': high_ball_ratio
                    }
                    
                    print(f"\n🎾 Frame {frame_idx}: 偵測到右手擊球！")
                    print(f"   類型: {shot_type}")
                    print(f"   Δy: {dy:.1f} pixels ({'往高處' if dy < 0 else '往低處'})")
                    print(f"   速度: {velocity:.1f} pixels/s")
                    print(f"   整體斜率(頭到尾): {overall_slope:.1f} pixels ({'下降' if overall_slope > 0 else '上升'})")
                    print(f"   垂直移動範圍: {y_range:.1f} pixels")
                    print(f"   最高點位置: {highest_position_ratio*100:.1f}% (在軌跡的前{highest_position_ratio*100:.0f}%)")
                    print(f"   右手臂抬起於 Frame {arm_raised_frame}，等待5幀後追蹤50幀")
                    print(f"   實際追蹤到 {len(tracking_points)} 個球位置點")
                    
                    # 不重置 arm_raised_frame 和 tracking_points，保留用於顯示
                else:
                    # ⚠️ 修正：如果找不到球，也要結束追蹤，避免無限循環
                    print(f"\n⚠️  Frame {frame_idx}: 追蹤55幀後無法找到有效的球位置")
                    print(f"   ball_at_raise: {ball_at_raise}, ball_now: {ball_now}")
                    print(f"   追蹤到 {len(tracking_points)} 個球位置點")
                    analysis_complete = True  # 標記完成，避免重複分析
                    # 清空追蹤狀態
                    arm_raised_frame = None
                    tracking_points = []
                    # ⭐ 新增：重置軌跡記錄器（追蹤失敗）
                    trajectory_logger.reset()
            
            arm_was_raised = current_arm_raised
        
        # 畫選中的人
        if selected_person:
            i = selected_person['index']
            x1i, y1i, x2i, y2i = selected_person['bbox']
            conf = selected_person['conf']
            bl = selected_person['bl']
            br = selected_person['br']
            in_court = selected_person['in_court']
            
            # 根據是否在場內選擇不同顏色
            color = (0, 255, 0) if in_court else (0, 165, 255)  # 綠色=場內，橘色=場外
            
            # ⭐ 隱藏人物框和文字標註
            # # 1) 畫人框 + 底部兩點
            # cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
            # cv2.circle(frame, bl, 3, color, -1)
            # cv2.circle(frame, br, 3, color, -1)
            
            # status_text = "IN COURT" if in_court else "OUT (Largest)"
            # text = f"person {conf:.2f} [{status_text}]"
            # text_pos = (x1i, max(0, y1i - 5))
            # cv2.putText(frame, text, text_pos,
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #             color, 2)
            
            # # 記錄文字區域（估計文字的 bounding box）
            # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            # text_x1 = text_pos[0]
            # text_y1 = text_pos[1] - text_size[1] - 5
            # text_x2 = text_pos[0] + text_size[0]
            # text_y2 = text_pos[1] + 5
            # text_boxes.append((text_x1, text_y1, text_x2, text_y2))

            # 2) 對選中的人畫 skeleton
            if kpts_all_cpu is not None:
                kpts = kpts_all_cpu[i]  # 使用預先轉換的版本
                draw_skeleton(frame, kpts, color=color)
        
        # === 顯示手臂狀態（明確顯示是右手臂）===
        if selected_person:
            arm_status_text = "RIGHT ARM: UP" if current_arm_raised else "RIGHT ARM: DOWN"
            arm_status_color = (0, 255, 0) if current_arm_raised else (128, 128, 128)
            cv2.putText(frame, arm_status_text,
                       (10, int(h - 40)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       arm_status_color, 2)
        
        # === 顯示追蹤中的球位置點（50幀追蹤，前5幀等待）===
        # ⚠️ 只在追蹤進行中顯示，分析完成後就不顯示了
        if arm_raised_frame is not None and not analysis_complete:
            frames_since_raise = frame_idx - arm_raised_frame
            
            # 顯示追蹤狀態文字（區分等待期和追蹤期）
            if frames_since_raise < 5:
                tracking_text = f"Waiting... ({frames_since_raise + 1}/5 frames)"
                text_color = (128, 128, 128)  # 灰色 = 等待中
            else:
                tracking_text = f"Tracking... ({len(tracking_points)}/50 frames)"
                text_color = (0, 255, 255)  # 青色 = 追蹤中
            
            cv2.putText(frame, tracking_text,
                       (10, int(h - 70)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       text_color, 2)
        
        # 只在有追蹤點時才顯示軌跡
        if arm_raised_frame is not None and len(tracking_points) > 0 and not analysis_complete:
            
            # 畫出所有追蹤點
            for i, (px, py, f_idx) in enumerate(tracking_points):
                # 根據時間順序使用漸變顏色（從藍色漸變到紅色）
                progress = i / max(len(tracking_points) - 1, 1)
                color_b = int(255 * (1 - progress))  # 藍色分量遞減
                color_r = int(255 * progress)        # 紅色分量遞增
                point_color = (color_b, 100, color_r)
                
                # 畫點（較大的圓圈）
                cv2.circle(frame, (int(px), int(py)), 5, point_color, -1)
                cv2.circle(frame, (int(px), int(py)), 6, (255, 255, 255), 1)  # 白色邊框
                
                # 每5個點標註一次幀數（避免太擁擠）
                if i % 5 == 0:
                    cv2.putText(frame, f"{i+1}", 
                               (int(px) + 8, int(py) - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                               (255, 255, 255), 1)
            
            # 畫軌跡連線（用漸變顏色）
            if len(tracking_points) > 1:
                for i in range(len(tracking_points) - 1):
                    pt1 = (int(tracking_points[i][0]), int(tracking_points[i][1]))
                    pt2 = (int(tracking_points[i+1][0]), int(tracking_points[i+1][1]))
                    
                    # 漸變顏色
                    progress = i / max(len(tracking_points) - 1, 1)
                    color_b = int(255 * (1 - progress))
                    color_r = int(255 * progress)
                    line_color = (color_b, 150, color_r)
                    
                    cv2.line(frame, pt1, pt2, line_color, 2)
            
            # 標示起點
            if len(tracking_points) > 0:
                start_pt = (int(tracking_points[0][0]), int(tracking_points[0][1]))
                cv2.circle(frame, start_pt, 8, (255, 0, 0), 2)  # 藍色圈 = 起點
                cv2.putText(frame, "START", 
                           (start_pt[0] + 10, start_pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 0, 0), 2)
            
            shot_display_timer -= 1
            
            # === 在畫面上畫出軌跡線（如果有資訊）===
            if trajectory_info and 'ball_at_shot' in trajectory_info and 'ball_after' in trajectory_info:
                pt0 = trajectory_info['ball_at_shot']
                pt1 = trajectory_info['ball_after']
                if pt0 and pt1:
                    # 畫軌跡線（綠色箭頭）
                    cv2.arrowedLine(frame, 
                                   (int(pt0[0]), int(pt0[1])), 
                                   (int(pt1[0]), int(pt1[1])), 
                                   (0, 255, 0), 3, tipLength=0.3)
                    # 標示起點和終點
                    cv2.circle(frame, (int(pt0[0]), int(pt0[1])), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (int(pt1[0]), int(pt1[1])), 6, (255, 0, 255), -1)
            
            # 當顯示計時器歸零時，清空所有追蹤相關的變數
            if shot_display_timer == 0:
                tracking_points = []
                arm_raised_frame = None
                analysis_complete = False
                trajectory_info = {}  # 清空軌跡資訊，避免殘留舊的箭頭

        # --- 2) 找球（YOLO 或 motion-based + 卡爾曼預測） ---
        # ⭐ 準備人的關鍵點資訊（用於排除手腕附近區域）
        current_person_keypoints = []
        if kpts_all_cpu is not None:
            current_person_keypoints = kpts_all_cpu  # 傳遞所有人的關鍵點
        
        # ⭐ 取得上一幀的球面積（用於面積穩定性檢查）
        last_ball_area = None
        if ball_bbox is not None:
            _, _, bw, bh = ball_bbox
            last_ball_area = bw * bh
        
        # ⭐⭐⭐ 整合 YOLO 羽球偵測 ⭐⭐⭐
        ball_pos = None
        ball_bbox = None
        cand_boxes = []
        
        if shuttlecock_model is not None and use_yolo_shuttlecock:
            # 使用 YOLO 偵測羽球
            shuttlecock_results = shuttlecock_model(
                frame,
                conf=SHUTTLECOCK_CONF,
                imgsz=640,
                verbose=False,
                device=device,
                half=True if device == 'mps' else False
            )[0]
            
            # ⭐ 顯示偵測資訊（每 30 幀）
            if frame_idx % 30 == 0 and len(shuttlecock_results.boxes) > 0:
                print(f"\n🔍 Frame {frame_idx}: 偵測到 {len(shuttlecock_results.boxes)} 個候選球")
                for idx, box in enumerate(shuttlecock_results.boxes):
                    conf = box.conf[0].cpu().numpy()
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    print(f"   候選 {idx+1}: conf={conf:.3f}, pos=({(x1+x2)/2:.0f}, {(y1+y2)/2:.0f})")
            
            # 處理 YOLO 偵測結果
            all_detections = []  # 儲存所有偵測結果用於顯示
            if len(shuttlecock_results.boxes) > 0:
                boxes = shuttlecock_results.boxes
                
                # ⭐ 收集所有偵測結果（包括被過濾的）用於調試
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    conf = box.conf[0].cpu().numpy()
                    all_detections.append({
                        'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                        'center': (cx, cy),
                        'conf': float(conf),
                        'filtered': False,
                        'reason': None
                    })
                
                # 找最佳候選球（離預測位置最近或面積最大）
                best_idx = None
                min_dist = 1e9
                
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    conf = box.conf[0].cpu().numpy()
                    
                    # 過濾條件
                    # 1. 在球場範圍內
                    if not (x_min_ball <= cx <= x_max_ball):
                        all_detections[idx]['filtered'] = True
                        all_detections[idx]['reason'] = 'out_of_court'
                        continue
                    # 2. 不在人身上
                    if any(point_in_box(cx, cy, pb) for pb in person_boxes):
                        all_detections[idx]['filtered'] = True
                        all_detections[idx]['reason'] = 'on_person'
                        continue
                    # 3. y 軸範圍限制
                    if cy < MIN_BALL_Y or cy > MAX_BALL_Y:
                        all_detections[idx]['filtered'] = True
                        all_detections[idx]['reason'] = f'y_range({cy:.0f})'
                        continue
                    
                    # 記錄候選框（用於後續顯示）
                    ball_w = x2 - x1
                    ball_h = y2 - y1
                    cand_boxes.append((int(x1), int(y1), int(ball_w), int(ball_h), cx, cy, ball_w*ball_h))
                    
                    # 選擇策略：優先選離預測位置最近的
                    if predicted_ball_pos is not None:
                        dist = np.hypot(cx - predicted_ball_pos[0], cy - predicted_ball_pos[1])
                    elif last_ball is not None:
                        dist = np.hypot(cx - last_ball[0], cy - last_ball[1])
                    else:
                        dist = 0  # 沒有參考，選第一個
                    
                    if best_idx is None or dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                
                # 設定最佳球位置
                if best_idx is not None:
                    box = boxes[best_idx]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    ball_pos = (cx, cy)
                    ball_bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
        
        # 如果 YOLO 沒偵測到，使用傳統差分方法作為備用
        if ball_pos is None:
            ball_pos, ball_bbox, gray, debug_vis, cand_boxes_motion = find_ball_motion(
                frame, prev_gray, last_ball,
                diff_thresh=BALL_DIFF_THRESH,
                bright_thresh=BALL_BRIGHT_THRESH,
                min_area=BALL_MIN_AREA,
                max_area=BALL_MAX_AREA,
                use_brightness=USE_BRIGHTNESS,
                predicted_pos=predicted_ball_pos,  # ⭐ 使用卡爾曼預測
                max_speed=MAX_BALL_SPEED,
                min_y=MIN_BALL_Y,
                max_y=MAX_BALL_Y,
                person_keypoints=current_person_keypoints,  # ⭐ 傳入人的關鍵點
                last_area=last_ball_area  # ⭐ 傳入上一幀球的面積
            )
            cand_boxes.extend(cand_boxes_motion)
        else:
            # YOLO 有偵測到，也需要更新 prev_gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        prev_gray = gray

        # === 記錄球的位置到 ball_history ===
        ball_info_text = []  # 用來顯示球的即時數據
        
        if ball_pos is not None:
            cx, cy = ball_pos
            # 只記錄在邊界內且不在人身上的球
            if (x_min_ball <= cx <= x_max_ball) and not any(
                point_in_box(cx, cy, pb) for pb in person_boxes
            ):
                ball_history.append((frame_idx, cx, cy))
                # ⭐ 新增：記錄到軌跡 logger
                trajectory_logger.add_point(frame_idx, cx, cy, detected=True)
                
                # === 計算球的即時速度和方向 ===
                if len(ball_history) >= 2:
                    # 找最近的兩個有效球位置
                    recent_balls = [b for b in list(ball_history)[-10:] if b is not None]
                    if len(recent_balls) >= 2:
                        prev_ball = recent_balls[-2]
                        curr_ball = recent_balls[-1]
                        
                        # 計算位移和速度
                        dx = curr_ball[1] - prev_ball[1]  # x 位移
                        dy = curr_ball[2] - prev_ball[2]  # y 位移
                        dt = (curr_ball[0] - prev_ball[0]) / fps  # 時間差
                        
                        if dt > 0:
                            velocity = np.hypot(dx, dy) / dt  # 速度 (px/s)
                            angle = np.degrees(np.arctan2(dy, abs(dx))) if dx != 0 else 0
                            
                            # 準備顯示的資訊
                            ball_info_text = [
                                f"Ball Pos: ({int(cx)}, {int(cy)})",
                                f"dx: {dx:.1f}  dy: {dy:.1f}",
                                f"Velocity: {velocity:.0f} px/s",
                                f"Angle: {angle:.1f} deg",
                                f"Direction: {'UP' if dy < 0 else 'DOWN'}"
                            ]
            else:
                ball_history.append(None)  # 球不符合條件
                # ⭐ 新增：記錄缺幀（球不符合條件）
                if trajectory_logger.get_trajectory_length() > 0:
                    trajectory_logger.add_point(frame_idx, None, None, detected=False)
        else:
            ball_history.append(None)  # 沒偵測到球
            # ⭐ 新增：記錄缺幀（沒偵測到球）
            if trajectory_logger.get_trajectory_length() > 0:
                trajectory_logger.add_point(frame_idx, None, None, detected=False)

        # === 選球策略：優先選右手附近的 candidate ===
        # 先畫「符合條件的 candidates」（藍色）：
        #   1. candidate 的中心 x 在 [x_min_ball, x_max_ball] 之間
        #   2. candidate 的中心不落在任何一個人的 bbox 裡
        #   3. candidate 的中心不落在文字標註區域裡
        filtered_candidates = []
        for (x_c, y_c, bw_c, bh_c, cx_c, cy_c, area_c) in cand_boxes:
            cx_c_i, cy_c_i = int(cx_c), int(cy_c)

            # (1) 限制在遠端左右兩線之間
            if not (x_min_ball <= cx_c <= x_max_ball):
                continue

            # (2) 不在任何人的 bounding box 內
            inside_any_person = False
            for pb in person_boxes:
                if point_in_box(cx_c, cy_c, pb):
                    inside_any_person = True
                    break
            if inside_any_person:
                continue
            
            # (3) 不在文字標註區域內
            inside_any_text = False
            for tb in text_boxes:
                if point_in_box(cx_c, cy_c, tb):
                    inside_any_text = True
                    break
            if inside_any_text:
                continue

            # 通過條件的 candidate
            filtered_candidates.append((x_c, y_c, bw_c, bh_c, cx_c, cy_c, area_c))

            # 畫藍色 candidate
            cv2.rectangle(frame, (x_c, y_c), (x_c + bw_c, y_c + bh_c), (255, 0, 0), 1)
            cv2.circle(frame, (cx_c_i, cy_c_i), 2, (255, 0, 0), -1)

        # === 智能選球策略：沿著斜率方向，選擇移動最遠但合理的候選球 ===
        selected_ball_pos = None
        selected_ball_bbox = None
        
        # 如果有符合條件的候選球
        if len(filtered_candidates) > 0:
            if predicted_pos_from_slope is not None and arm_raised_frame is not None and not analysis_complete and len(tracking_points) >= 2:
                # 追蹤進行中：使用斜率和歷史速度來選球
                pred_x, pred_y = predicted_pos_from_slope
                
                # 計算歷史平均速度範圍（用於判斷是否合理）
                recent_speeds = []
                for i in range(max(0, len(tracking_points) - 10), len(tracking_points) - 1):
                    dx = tracking_points[i+1][0] - tracking_points[i][0]
                    dy = tracking_points[i+1][1] - tracking_points[i][1]
                    speed = np.hypot(dx, dy)
                    recent_speeds.append(speed)
                
                if len(recent_speeds) > 0:
                    avg_speed = np.mean(recent_speeds)
                    max_reasonable_speed = avg_speed * 2.5  # 允許速度波動到2.5倍
                    min_reasonable_speed = avg_speed * 0.3  # 允許減速到0.3倍
                else:
                    max_reasonable_speed = 150  # 預設最大速度
                    min_reasonable_speed = 5    # 預設最小速度
                
                # 計算每個候選球沿著預測方向的「投影距離」和「實際移動距離」
                last_x, last_y = tracking_points[-1][0], tracking_points[-1][1]
                
                # 預測方向向量（單位化）
                pred_dx = pred_x - last_x
                pred_dy = pred_y - last_y
                pred_norm = np.hypot(pred_dx, pred_dy)
                
                if pred_norm > 0:
                    pred_dir_x = pred_dx / pred_norm
                    pred_dir_y = pred_dy / pred_norm
                else:
                    pred_dir_x, pred_dir_y = 0, 0
                
                best_projection = -1e9  # 最大投影距離
                
                for (x_c, y_c, bw_c, bh_c, cx_c, cy_c, area_c) in filtered_candidates:
                    # 計算候選球相對於上一幀的位移
                    dx = cx_c - last_x
                    dy = cy_c - last_y
                    actual_distance = np.hypot(dx, dy)
                    
                    # 檢查移動距離是否在合理範圍內
                    if actual_distance < min_reasonable_speed or actual_distance > max_reasonable_speed:
                        continue  # 跳過不合理的移動距離
                    
                    # 計算在預測方向上的投影距離（點積）
                    projection = dx * pred_dir_x + dy * pred_dir_y
                    
                    # 選擇沿著預測方向投影最遠的候選球
                    if projection > best_projection:
                        best_projection = projection
                        selected_ball_pos = (cx_c, cy_c)
                        selected_ball_bbox = (x_c, y_c, bw_c, bh_c)
                
                # 如果沒有找到合理的候選，fallback 到選最高的
                if selected_ball_pos is None:
                    min_y = 1e9
                    for (x_c, y_c, bw_c, bh_c, cx_c, cy_c, area_c) in filtered_candidates:
                        if cy_c < min_y:
                            min_y = cy_c
                            selected_ball_pos = (cx_c, cy_c)
                            selected_ball_bbox = (x_c, y_c, bw_c, bh_c)
            else:
                # 沒有追蹤進行中：直接選 y 座標最小的（畫面最高的）
                min_y = 1e9
                for (x_c, y_c, bw_c, bh_c, cx_c, cy_c, area_c) in filtered_candidates:
                    if cy_c < min_y:
                        min_y = cy_c
                        selected_ball_pos = (cx_c, cy_c)
                        selected_ball_bbox = (x_c, y_c, bw_c, bh_c)
        
        # 如果沒有找到候選球，fallback 到原本的邏輯（找離上一幀最近的）
        if selected_ball_pos is None and ball_pos is not None:
            cx, cy = ball_pos
            # 檢查是否在邊界內且不在人身上
            if (x_min_ball <= cx <= x_max_ball) and not any(
                point_in_box(cx, cy, pb) for pb in person_boxes
            ):
                selected_ball_pos = ball_pos
                selected_ball_bbox = ball_bbox
        
        # 畫球（紅色）
        if selected_ball_pos is not None and selected_ball_bbox is not None:
            cx, cy = selected_ball_pos
            x, y, bw, bh = selected_ball_bbox
            
            # ⭐ 更新卡爾曼濾波器（計算速度和預測下一幀位置）
            if last_ball is not None:
                # 計算速度向量
                vx = cx - last_ball[0]
                vy = cy - last_ball[1]
                
                # 平滑速度（指數移動平均）
                if ball_velocity is None:
                    ball_velocity = (vx, vy)
                else:
                    alpha = 0.7  # 平滑係數（越大越信任新速度）
                    ball_velocity = (
                        alpha * vx + (1 - alpha) * ball_velocity[0],
                        alpha * vy + (1 - alpha) * ball_velocity[1]
                    )
                
                # 預測下一幀位置
                predicted_ball_pos = (
                    cx + ball_velocity[0],
                    cy + ball_velocity[1]
                )
                
                # ⭐ 視覺化預測位置（綠色虛線圈）
                pred_x, pred_y = int(predicted_ball_pos[0]), int(predicted_ball_pos[1])
                cv2.circle(frame, (pred_x, pred_y), 6, (0, 255, 0), 1)  # 預測位置
                cv2.line(frame, (int(cx), int(cy)), (pred_x, pred_y), (0, 255, 0), 1)  # 速度向量
            else:
                ball_velocity = None
                predicted_ball_pos = None
            
            last_ball = selected_ball_pos  # 更新 last_ball 用於下一幀追蹤
            
            # 畫實際偵測到的球（紅色）
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            
            # 更新 ball_pos 供後續使用（記錄到 history）
            ball_pos = selected_ball_pos
            ball_bbox = selected_ball_bbox
        else:
            # ⭐ 如果丟失球，清空速度預測（避免持續錯誤預測）
            if ball_velocity is not None:
                # 允許短暫丟失（5幀內），保留預測
                # 超過5幀就清空
                pass  # 可以後續改進
            predicted_ball_pos = None
        
        # === 在追蹤期間收集球的位置點（移到這裡，確保球已經被偵測） ===
        # ⭐ 只在手臂抬起後第 5-55 幀（共50幀）收集球位置，前5幀跳過
        if arm_raised_frame is not None and not analysis_complete:
            frames_since_raise = frame_idx - arm_raised_frame
            
            # 只在第 5-55 幀收集球位置（跳過前5幀，收集50幀）
            if frames_since_raise >= 5:
                # === 計算中段斜率並用於校正軌跡 ===
                # 當有足夠的中段數據（第20-35幀）時，計算預測軌跡
                predicted_pos_from_slope = None
                
                if len(tracking_points) >= 20:  # 至少有20個點才能計算中段斜率
                    # 取中段15幀（第20-35幀或已有的點）來計算斜率
                    mid_start_idx = 19  # 第20幀（索引19）
                    mid_end_idx = min(34, len(tracking_points) - 1)  # 第35幀或最後一幀
                    
                    if mid_end_idx - mid_start_idx >= 10:  # 確保有足夠跨度
                        # 計算中段的平均速度向量
                        x_start, y_start = tracking_points[mid_start_idx][0], tracking_points[mid_start_idx][1]
                        x_end, y_end = tracking_points[mid_end_idx][0], tracking_points[mid_end_idx][1]
                        
                        # 時間跨度（幀數）
                        frame_span = mid_end_idx - mid_start_idx
                        
                        # 計算每幀的平均位移
                        avg_vx = (x_end - x_start) / frame_span
                        avg_vy = (y_end - y_start) / frame_span
                        
                        # 預測下一幀位置（基於最後一個追蹤點）
                        if len(tracking_points) > 0:
                            last_x, last_y = tracking_points[-1][0], tracking_points[-1][1]
                            predicted_pos_from_slope = (last_x + avg_vx, last_y + avg_vy)
                
                # === 收集球位置或使用預測位置 ===
                if selected_ball_pos is not None:
                    # 有偵測到球，直接使用
                    tracking_points.append((selected_ball_pos[0], selected_ball_pos[1], frame_idx))
                    print(f"   Frame {frame_idx} (追蹤第 {len(tracking_points)}/50 幀): 球位置 ({int(selected_ball_pos[0])}, {int(selected_ball_pos[1])}) [偵測]")
                elif predicted_pos_from_slope is not None:
                    # 偵測失敗，使用中段斜率預測的位置
                    pred_x, pred_y = predicted_pos_from_slope
                    tracking_points.append((pred_x, pred_y, frame_idx))
                    print(f"   Frame {frame_idx} (追蹤第 {len(tracking_points)}/50 幀): 球位置 ({int(pred_x)}, {int(pred_y)}) [預測]")
                else:
                    # 沒有偵測也沒有足夠數據預測，跳過此幀
                    print(f"   Frame {frame_idx} (追蹤第 {len(tracking_points)}/50 幀): 未偵測到球，資料不足無法預測")
            elif frames_since_raise < 5:
                print(f"   Frame {frame_idx} (等待期 {frames_since_raise+1}/5): 跳過...")

        # === 即時顯示資訊（左上角） ===
        # 只顯示球的即時數據，擊球結果移到畫面下方
        display_info = []
        
        if ball_info_text:
            # 顯示球的即時數據
            display_info = ball_info_text
        
        if display_info:
            # 半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 10 + 28 * len(display_info)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 顯示每一行資訊
            for idx, text in enumerate(display_info):
                y_pos = 30 + idx * 23
                # 球位置資訊用青色
                text_color = (0, 255, 255)
                cv2.putText(frame, text, (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)
        
        # === 顯示幀數（右上角） ===
        frame_text = f"Frame: {frame_idx}/{total_frames}"
        cv2.putText(frame, frame_text, (int(w - 250), 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # === ⭐ 顯示擊球偵測結果（畫面上方中央，最後繪製確保不被覆蓋）===
        if shot_display_timer > 0:
            # 只有當 trajectory_info 存在時才顯示
            if trajectory_info:
                # ⭐ 計算顯示位置（畫面上方中央）
                panel_width = 500
                panel_x = (w - panel_width) // 2  # 中央位置
                panel_y = 60  # 從上方 60px 開始（避開其他資訊）
                
                # 顯示擊球資訊
                vel = trajectory_info.get('velocity', 0)
                dy_total = trajectory_info.get('dy', 0)
                high_ratio = trajectory_info.get('high_ball_ratio', 0)
                
                # 計算頭尾 y 變化
                head_to_tail_dy = 0
                if len(tracking_points) >= 2:
                    head_to_tail_dy = tracking_points[-1][1] - tracking_points[0][1]
                
                shot_info = [
                    f"SHOT DETECTED: {shot_type}",
                    f"Delta-Y: {dy_total:.1f} px ({'UP' if dy_total < 0 else 'DOWN'})",
                    f"Velocity: {vel:.1f} px/s",
                    f"High Ball Ratio: {high_ratio:.1%}",
                    f"Head-Tail Y: {head_to_tail_dy:.1f} px",
                    f"Tracked Points: {len(tracking_points)}"
                ]
                
                line_height = 30
                panel_height = 20 + len(shot_info) * line_height
                
                # 繪製半透明深色背景
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), 
                            (panel_x + panel_width, panel_y + panel_height), 
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                
                # 顯示每一行（醒目的黃色文字，較大字體）
                for idx, text in enumerate(shot_info):
                    y_pos = panel_y + 25 + idx * line_height
                    # 第一行（SHOT DETECTED）用更大更亮的顏色
                    if idx == 0:
                        cv2.putText(frame, text, (panel_x + 15, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                    else:
                        cv2.putText(frame, text, (panel_x + 15, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            
            shot_display_timer -= 1

        # ⭐ 顯示所有 YOLO 偵測結果（統一在畫面上方中央）
        if SHOW_ALL_DETECTIONS and 'all_detections' in locals() and all_detections:
            # 只顯示黃線內的候選球
            valid_detections = [det for det in all_detections 
                              if x_min_ball <= det['center'][0] <= x_max_ball]
            
            if valid_detections:
                # 在畫面上方中央繪製偵測資訊面板
                panel_width = 400
                panel_x = (w - panel_width) // 2  # 中央位置
                panel_y = 10
                line_height = 22
                panel_height = 30 + len(valid_detections) * line_height
                
                # 繪製半透明背景
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), 
                            (panel_x + panel_width, panel_y + panel_height), 
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                
                # 標題
                cv2.putText(frame, f"Ball Detections ({len(valid_detections)})", 
                           (panel_x + 10, panel_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 顯示每個偵測結果
                for idx, det in enumerate(valid_detections):
                    ball_x, ball_y, ball_w, ball_h = det['bbox']
                    cx, cy = det['center']
                    conf = det['conf']
                    filtered = det['filtered']
                    reason = det['reason']
                    
                    # 被過濾的用橘色，未過濾的用綠色
                    color = (0, 165, 255) if filtered else (0, 255, 0)
                    
                    # 顯示文字資訊
                    status = f"FILTERED ({reason})" if filtered else "SELECTED"
                    label = f"{idx+1}. ({int(cx)},{int(cy)}) conf:{conf:.2f} [{status}]"
                    
                    y_pos = panel_y + 45 + idx * line_height
                    cv2.putText(frame, label, (panel_x + 15, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    
                    # 在球位置畫對應序號的小圓點
                    cv2.circle(frame, (int(cx), int(cy)), 8, color, 2)
                    cv2.putText(frame, str(idx+1), (int(cx)-5, int(cy)+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- 3) 寫出 + 即時顯示 ---
        out.write(frame)
        
        # 即時顯示處理結果
        cv2.imshow('Badminton Analysis', frame)
        
        # 按 'q' 可以提前結束，按 'p' 暫停
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⚠️  使用者中斷處理")
            break
        elif key == ord('p'):
            print("\n⏸️  暫停中，按任意鍵繼續...")
            cv2.waitKey(0)

        frame_idx += 1
        
        # === 進度顯示（每10秒） ===
        current_time = time.time()
        if current_time - last_progress_time >= 10.0:
            elapsed_time = current_time - start_time
            progress_percent = (frame_idx / total_frames) * 100
            fps_current = frame_idx / elapsed_time
            remaining_frames = total_frames - frame_idx
            eta_seconds = remaining_frames / fps_current if fps_current > 0 else 0
            
            print(f"進度: {frame_idx}/{total_frames} ({progress_percent:.1f}%) | "
                  f"已耗時: {elapsed_time:.1f}秒 | "
                  f"處理速度: {fps_current:.1f} fps | "
                  f"預估剩餘: {eta_seconds:.1f}秒")
            
            last_progress_time = current_time

    cap.release()
    out.release()
    cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
    
    # === 最終統計 ===
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    
    print("\n" + "="*60)
    print("處理完成！")
    print("="*60)
    print(f"總幀數: {frame_idx}")
    print(f"總耗時: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
    print(f"平均處理速度: {avg_fps:.1f} fps")
    print(f"輸出檔案: {OUTPUT_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
