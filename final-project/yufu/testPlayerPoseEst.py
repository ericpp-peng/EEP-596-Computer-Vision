import cv2
import numpy as np
import torch
import torch.serialization
from ultralytics import YOLO
import time
from collections import deque

# Fix for PyTorch 2.6+ weights_only loading issue
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.PoseModel'])

# ===================== 參數區：請依你的環境修改 =====================
# 影片路徑
VIDEO_PATH = "./20250711_short.mp4"

# 輸出影片路徑
OUTPUT_PATH = "./output_with_pose.mp4"

# 球場標註座標 (np.save 出來的 court_pts.npy)
COURT_PTS_PATH = "./court_pts.npy"

# ⭐ 羽球邊界線座標（可選，如果有的話會用手動標註的，沒有就自動計算）
BALL_BOUNDARY_PATH = "./ball_boundary.npy"

# ⭐ 改成 YOLO Pose 權重（人 + skeleton）
YOLO_WEIGHTS = "./yolov8n-pose.pt"

# 球偵測參數（請用你在滑桿小工具上調好的那組數值）
BALL_DIFF_THRESH = 72
BALL_BRIGHT_THRESH = 255   # 如果 use_brightness=False，這個只影響 debug，不影響 candidate
BALL_MIN_AREA = 5
BALL_MAX_AREA = 95
USE_BRIGHTNESS = False     # ⭐ False = 只看差分（比較接近你 slider 的直覺）
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
                     diff_thresh=25,      # 差分閾值
                     bright_thresh=200,   # 亮度閾值 (0-255)
                     min_area=5, max_area=400,
                     use_brightness=True):
    """
    利用前後幀差分 +（可選）亮度 + 面積偵測羽球
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

        # 面積過小或過大都不要
        if area < min_area or area > max_area:
            continue

        candidate_boxes.append((x, y, w, h, cx, cy, area))

        if last_pos is None:
            best_pos = (cx, cy)
            best_bbox = (x, y, w, h)
        else:
            dist = np.hypot(cx - last_pos[0], cy - last_pos[1])
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


def is_arm_raised(kpts):
    """
    判斷手臂是否抬起（右手腕是否高於右肩）
    Returns: True 如果手臂抬起，False 否則
    """
    if len(kpts) < 17:
        return False
    
    r_shoulder = kpts[6]  # 右肩
    r_wrist = kpts[10]    # 右手腕
    
    if r_shoulder[0] <= 0 or r_wrist[0] <= 0:
        return False
    
    # 手腕 y 座標小於肩膀（y 軸向下為正）表示手腕在肩膀上方
    return r_wrist[1] < (r_shoulder[1] - 30)  # 至少高於肩膀 30 pixels


def classify_shot_by_trajectory(ball_history, shot_frame_idx, fps=30, 
                                  frames_after=5, frames_before=3):
    """
    根據球的軌跡判斷擊球類型
    
    Args:
        ball_history: deque of (frame_idx, cx, cy) or None
        shot_frame_idx: 擊球發生的幀數
        fps: 影片 fps
        frames_after: 擊球後取幾幀來計算
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
    print(f"載入模型: {YOLO_WEIGHTS}")
    people_model = YOLO(YOLO_WEIGHTS)
    people_model.to(device)  # 將模型移到 GPU
    
    # 啟用 FP16 半精度運算（MPS 支援，可大幅加速）
    if device == 'mps':
        print("啟用 FP16 半精度運算以加速推論")
    
    print(f"模型已載入到: {device}")

    prev_gray = None
    last_ball = None
    frame_idx = 0

    # === 球軌跡記錄（用於判斷擊球類型）===
    ball_history = deque(maxlen=60)  # 保留最近 60 幀的球位置（2秒@30fps）

    # === 手臂動作追蹤變數 ===
    arm_was_raised = False      # 上一幀手臂是否抬起
    arm_raised_frame = None     # 手臂抬起的幀數（用於追蹤後續球的運動）
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
        cv2.line(frame, (x_min_ball, 0), (x_min_ball, h - 1), (0, 255, 255), 1)
        cv2.line(frame, (x_max_ball, 0), (x_max_ball, h - 1), (0, 255, 255), 1)

        # --- 1) YOLO Pose 抓「人」 + skeleton ---
        person_boxes = []  # 存所有人的 bbox（不管在不在場內，用來過濾球）
        text_boxes = []    # 存所有文字標註的 bbox（用來過濾球的候選區域）

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
            
            # 記錄手臂抬起的時刻
            if current_arm_raised and not arm_was_raised:
                # 手臂從放下變成抬起
                arm_raised_frame = frame_idx
                tracking_points = []  # 清空追蹤點
                analysis_complete = False  # 重置分析狀態
                print(f"\n📍 Frame {frame_idx}: 手臂抬起，開始追蹤球的運動...")
            
            # ⚠️ 注意：tracking_points 的收集移到偵測球之後（在後面的代碼中）
            
            # 在手臂抬起後約30幀（1秒@30fps）檢查球的運動方向
            if arm_raised_frame is not None and not analysis_complete and (frame_idx - arm_raised_frame) >= 30:
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
                    # 計算 y 軸方向變化
                    dy = ball_now[1] - ball_at_raise[1]
                    dx = ball_now[0] - ball_at_raise[0]
                    
                    # 計算速度
                    dt = (frame_idx - arm_raised_frame) / fps
                    distance = np.hypot(dx, dy)
                    velocity = distance / dt if dt > 0 else 0
                    
                    # 根據 y 軸方向判斷擊球類型
                    if dy < -20:  # y 變小（往上 = 往高處）
                        shot_type = "Clear (高遠球)"
                    elif dy > 20:  # y 變大（往下 = 往低處）
                        if velocity > 500:  # 速度快
                            shot_type = "Smash (殺球)"
                        else:  # 速度慢
                            shot_type = "Drop (切球)"
                    else:
                        shot_type = "Drive (平抽)"
                    
                    shot_detected = True
                    shot_display_timer = 60  # 顯示 60 幀（2秒）
                    analysis_complete = True  # 標記分析完成
                    
                    trajectory_info = {
                        'dx': dx,
                        'dy': dy,
                        'velocity': velocity,
                        'ball_at_raise': ball_at_raise,
                        'ball_now': ball_now
                    }
                    
                    print(f"\n🎾 Frame {frame_idx}: 偵測到擊球！")
                    print(f"   類型: {shot_type}")
                    print(f"   Δy: {dy:.1f} pixels ({'往高處' if dy < 0 else '往低處'})")
                    print(f"   速度: {velocity:.1f} pixels/s")
                    print(f"   手臂抬起於 Frame {arm_raised_frame}，已追蹤 {frame_idx - arm_raised_frame} 幀")
                    print(f"   追蹤到 {len(tracking_points)} 個球位置點")
                    
                    # 不重置 arm_raised_frame 和 tracking_points，保留用於顯示
                else:
                    # ⚠️ 修正：如果找不到球，也要結束追蹤，避免無限循環
                    print(f"\n⚠️  Frame {frame_idx}: 追蹤30幀後無法找到有效的球位置")
                    print(f"   ball_at_raise: {ball_at_raise}, ball_now: {ball_now}")
                    print(f"   追蹤到 {len(tracking_points)} 個球位置點")
                    analysis_complete = True  # 標記完成，避免重複分析
                    # 清空追蹤狀態
                    arm_raised_frame = None
                    tracking_points = []
            
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
        
        # === 顯示手臂狀態 ===
        if selected_person:
            arm_status_text = "ARM: UP" if current_arm_raised else "ARM: DOWN"
            arm_status_color = (0, 255, 0) if current_arm_raised else (128, 128, 128)
            cv2.putText(frame, arm_status_text,
                       (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       arm_status_color, 2)
        
        # === 顯示追蹤中的球位置點（30幀判斷點）===
        if arm_raised_frame is not None and len(tracking_points) > 0:
            # 顯示追蹤狀態文字
            if not analysis_complete:
                tracking_text = f"Tracking... ({len(tracking_points)}/30 frames)"
                cv2.putText(frame, tracking_text,
                           (10, h - 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 255), 2)
            
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
            
            # 標示起點和終點
            if len(tracking_points) > 0:
                start_pt = (int(tracking_points[0][0]), int(tracking_points[0][1]))
                cv2.circle(frame, start_pt, 8, (255, 0, 0), 2)  # 藍色圈 = 起點
                cv2.putText(frame, "START", 
                           (start_pt[0] + 10, start_pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 0, 0), 2)
                
                if analysis_complete and len(tracking_points) > 1:
                    end_pt = (int(tracking_points[-1][0]), int(tracking_points[-1][1]))
                    cv2.circle(frame, end_pt, 8, (0, 0, 255), 2)  # 紅色圈 = 終點
                    cv2.putText(frame, "END", 
                               (end_pt[0] + 10, end_pt[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (0, 0, 255), 2)
        
        # === 顯示擊球偵測結果 ===
        if shot_display_timer > 0:
            # 半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2 - 250, 50), (w//2 + 250, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # 主要文字（橘色）
            cv2.putText(frame, "SHOT DETECTED!", 
                       (w//2 - 220, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            
            # 擊球類型（黃色，更大）
            cv2.putText(frame, f"Type: {shot_type}", 
                       (w//2 - 180, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # 顯示軌跡資訊（小字，青色）
            if trajectory_info:
                vel = trajectory_info.get('velocity', 0)
                angle = trajectory_info.get('angle', 0)
                info_text = f"V: {vel:.0f} px/s  Angle: {angle:.1f}deg"
                cv2.putText(frame, info_text, 
                           (w//2 - 180, 175),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
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
            
            shot_display_timer -= 1
            
            # 當顯示計時器歸零時，清空追蹤點
            if shot_display_timer == 0:
                tracking_points = []
                arm_raised_frame = None
                analysis_complete = False

        # --- 2) 找球（motion-based） ---
        ball_pos, ball_bbox, gray, debug_vis, cand_boxes = find_ball_motion(
            frame, prev_gray, last_ball,
            diff_thresh=BALL_DIFF_THRESH,
            bright_thresh=BALL_BRIGHT_THRESH,
            min_area=BALL_MIN_AREA,
            max_area=BALL_MAX_AREA,
            use_brightness=USE_BRIGHTNESS
        )
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
        else:
            ball_history.append(None)  # 沒偵測到球

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

        # === 新策略：選擇 y 最小（畫面最高）的候選球 ===
        selected_ball_pos = None
        selected_ball_bbox = None
        
        # 如果有符合條件的候選球，選擇 y 座標最小的（畫面最高的）
        if len(filtered_candidates) > 0:
            # 找 y 座標最小的 candidate
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
            last_ball = selected_ball_pos  # 更新 last_ball 用於下一幀追蹤
            
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            
            # 更新 ball_pos 供後續使用（記錄到 history）
            ball_pos = selected_ball_pos
            ball_bbox = selected_ball_bbox
        
        # === 在追蹤期間收集球的位置點（移到這裡，確保球已經被偵測） ===
        if arm_raised_frame is not None and not analysis_complete:
            # 檢查當前幀是否有球
            if selected_ball_pos is not None:
                tracking_points.append((selected_ball_pos[0], selected_ball_pos[1], frame_idx))
                print(f"   Frame {frame_idx}: 追蹤球位置 ({int(selected_ball_pos[0])}, {int(selected_ball_pos[1])})")

        # === 即時顯示球的數據（左上角） ===
        if ball_info_text:
            # 半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 10 + 30 * len(ball_info_text)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 顯示每一行資訊
            for idx, text in enumerate(ball_info_text):
                y_pos = 35 + idx * 25
                cv2.putText(frame, text, (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # === 顯示幀數（右上角） ===
        frame_text = f"Frame: {frame_idx}/{total_frames}"
        cv2.putText(frame, frame_text, (w - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
