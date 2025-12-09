import cv2
import numpy as np
import torch
import torch.serialization
from ultralytics import YOLO
import time

# Fix for PyTorch 2.6+ weights_only loading issue
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.PoseModel'])

# ===================== 參數區：請依你的環境修改 =====================
# 影片路徑
VIDEO_PATH = "./Chou-pro.mp4"

# 輸出影片路徑
OUTPUT_PATH = "./output_chou_pro.mp4"

# 球場標註座標 (np.save 出來的 court_pts.npy)
COURT_PTS_PATH = "./court_pts_pro.npy"

# ⭐ 羽球邊界線座標（可選，如果有的話會用手動標註的，沒有就自動計算）
BALL_BOUNDARY_PATH = "./ball_boundary_pro.npy"

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

    # === 手臂動作追蹤變數 ===
    arm_was_raised = False      # 上一幀手臂是否抬起
    shot_detected = False       # 是否偵測到擊球動作
    shot_type = ""             # 擊球類型（暫定為 "Smash/Clear/Drop"）
    shot_display_timer = 0      # 顯示計時器（顯示 30 幀 = 約 1 秒）

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
            
            # 偵測「抬起 → 放下」的動作
            if arm_was_raised and not current_arm_raised:
                # 手臂從抬起變成放下 = 可能是擊球動作
                shot_detected = True
                shot_type = "Smash / Clear / Drop"
                shot_display_timer = 30  # 顯示 30 幀
                # print(f"🎾 Frame {frame_idx}: 偵測到擊球動作！")  # 註解掉，避免過多輸出
            
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
            
            # 1) 畫人框 + 底部兩點
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
            cv2.circle(frame, bl, 3, color, -1)
            cv2.circle(frame, br, 3, color, -1)
            
            status_text = "IN COURT" if in_court else "OUT (Largest)"
            cv2.putText(frame, f"person {conf:.2f} [{status_text}]",
                        (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2)

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
        
        # === 顯示擊球偵測結果 ===
        if shot_display_timer > 0:
            # 半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2 - 250, 50), (w//2 + 250, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # 主要文字（橘色）
            cv2.putText(frame, "SHOT DETECTED!", 
                       (w//2 - 220, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            cv2.putText(frame, shot_type, 
                       (w//2 - 180, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            shot_display_timer -= 1

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

        # 先畫「符合條件的 candidates」（藍色）：
        #   1. candidate 的中心 x 在 [x_min_ball, x_max_ball] 之間
        #   2. candidate 的中心不落在任何一個人的 bbox 裡
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

            # 通過條件的 candidate
            filtered_candidates.append((x_c, y_c, bw_c, bh_c, cx_c, cy_c, area_c))

            # 畫藍色 candidate
            cv2.rectangle(frame, (x_c, y_c), (x_c + bw_c, y_c + bh_c), (255, 0, 0), 1)
            cv2.circle(frame, (cx_c_i, cy_c_i), 2, (255, 0, 0), -1)

        # 再畫球（紅色）：這裡只要畫就好，不再分 IN/OUT
        if ball_pos is not None and ball_bbox is not None:
            cx, cy = ball_pos
            x, y, bw, bh = ball_bbox
            last_ball = ball_pos

            # 如果球中心 x 不在邊界內，或落在人身上，就不畫
            if (x_min_ball <= cx <= x_max_ball) and not any(
                point_in_box(cx, cy, pb) for pb in person_boxes
            ):
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        # --- 3) 寫出（不顯示） ---
        out.write(frame)

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
