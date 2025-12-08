from ultralytics import YOLO
import cv2
import numpy as np
import time


# =============================
# 1. --- 關節角度 & 動作分類函式 ---
# =============================

def angle(a, b, c):
    """Compute angle ABC (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def classify_action(kpts_prev, kpts_now):
    """Rule-based 羽球動作分類"""

    # 右手羽球動作（右肩、右肘、右腕）
    R_sh = kpts_now[6]
    R_el = kpts_now[8]
    R_wr = kpts_now[10]

    # 髖（腳步判斷）
    R_hip = kpts_now[12]

    # ---- 特徵 ----
    elbow_angle = angle(R_sh, R_el, R_wr)
    shoulder_angle = angle(R_el, R_sh, R_hip)

    if kpts_prev is not None:
        wrist_vel = np.linalg.norm(R_wr - kpts_prev[10])
        hip_vel   = np.linalg.norm(R_hip - kpts_prev[12])
    else:
        wrist_vel = 0
        hip_vel   = 0

    # ---- 規則分類 ----

    # 腳步移動（身體移動最快）
    if hip_vel > 15:
        return "FOOTWORK"

    # 殺球：手腕速度快 + 手臂抬高 + 肘伸直
    if shoulder_angle > 70 and wrist_vel > 15 and elbow_angle < 30:
        return "SMASH"

    # 切球：中速揮拍、肘角度較大但不揮直
    if 3 < wrist_vel < 12 and elbow_angle > 40:
        return "DROP"

    # 平抽：低肩角 + 高手腕速度
    if shoulder_angle < 50 and wrist_vel > 10:
        return "DRIVE"

    # 放小球：動作很輕、小、慢
    if wrist_vel < 2 and shoulder_angle < 30:
        return "NET"

    return "UNKNOWN"


# =============================
# 2. --- YOLO Models ---
# =============================

det_model  = YOLO("yolov8n.pt", verbose=False)
pose_model = YOLO("yolov8n-pose.pt", verbose=False)

cap = cv2.VideoCapture("20250711_short.mp4")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- 輸出影片 ---
out = cv2.VideoWriter(
    "output_action.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

prev_kpts = None
frame_idx = 0

start_time = time.time()
last_report = start_time


# =============================
# 3. --- Main Loop ---
# =============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==== YOLO detect：找球員 ====
    det_results = det_model(frame, imgsz=320, verbose=False)[0]

    h_img, w_img, _ = frame.shape
    center = np.array([w_img/2, h_img/2])

    best_score = -1
    best_box = None

    for box in det_results.boxes:
        if int(box.cls[0]) != 0:  # only person
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2-x1)*(y2-y1)
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        dist = np.linalg.norm(np.array([cx,cy]) - center)

        score = area - dist * 50
        if score > best_score:
            best_score = score
            best_box = (x1,y1,x2,y2)

    if best_box is None:
        out.write(frame)
        prev_kpts = None
        continue

    x1,y1,x2,y2 = best_box
    crop = frame[y1:y2, x1:x2]

    # ==== YOLO Pose ====
    pose_results = pose_model(crop, imgsz=320, verbose=False)[0]

    action = "UNKNOWN"

    if len(pose_results.keypoints) > 0:
        kpts = pose_results.keypoints.xy[0].cpu().numpy()

        # --- 動作分類 ---
        action = classify_action(prev_kpts, kpts)
        prev_kpts = kpts

        # --- 畫骨架點 ---
        for (px, py) in kpts:
            px = int(px + x1)
            py = int(py + y1)
            cv2.circle(frame, (px, py), 3, (0,255,0), -1)

    # ==== 畫動作文字 ====
    cv2.putText(
        frame, action,
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0,0,255), 3
    )

    # ==== 寫入輸出影片 ====
    out.write(frame)

    # ==== 進度條（每 30 秒顯示一次）====
    frame_idx += 1
    now = time.time()
    if now - last_report >= 30:
        percent = frame_idx / total_frames * 100
        print(f"Progress: {percent:.2f}%  ({frame_idx}/{total_frames})")
        last_report = now


cap.release()
out.release()

print("\n🎉 Done! 影片已輸出：output_action.mp4\n")
