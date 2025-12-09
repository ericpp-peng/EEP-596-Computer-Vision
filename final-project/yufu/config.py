"""
羽球分析系統 - 統一配置檔
整合所有參數設定，方便調整和管理
"""

# ===================== 影片與檔案路徑 =====================
VIDEO_PATH = "./20250711_short.mp4"
OUTPUT_PATH = "./output_with_pose.mp4"
COURT_PTS_PATH = "./court_pts.npy"
BALL_BOUNDARY_PATH = "./ball_boundary.npy"
SHOT_LOG_FILE = "./shot_annotations.json"

# ===================== YOLO 模型設定 =====================
# 人物姿態偵測模型
YOLO_POSE_WEIGHTS = "./yolov8n-pose.pt"

# 羽球偵測模型
SHUTTLECOCK_WEIGHTS = "./runs/detect/shuttlecock_improved_20251209_122742/weights/best.pt"
USE_YOLO_SHUTTLECOCK = True  # 是否使用 YOLO 羽球偵測
SHUTTLECOCK_CONF = 0.15  # YOLO 偵測信心度門檻
SHOW_ALL_DETECTIONS = True  # 顯示所有偵測結果（包括被過濾的）

# ===================== 球偵測參數（傳統差分法）=====================
BALL_DIFF_THRESH = 40      # 差分閾值（越小越敏感）
BALL_BRIGHT_THRESH = 180   # 亮度閾值
BALL_MIN_AREA = 2          # 最小面積
BALL_MAX_AREA = 120        # 最大面積
USE_BRIGHTNESS = False     # 是否使用亮度過濾

# ===================== 球運動限制參數 =====================
MAX_BALL_SPEED = 2000      # 最大速度限制 (px/frame)
MAX_BALL_Y = 650           # 最大 y 座標
MIN_BALL_Y = 30            # 最小 y 座標
BALL_PREDICTION_WEIGHT = 0.3  # 卡爾曼預測權重

# ===================== 球拍排除參數 =====================
WRIST_EXCLUSION_RADIUS = 100  # 排除手腕周圍的半徑（像素）
RACKET_SHAPE_ASPECT_RATIO = 2.5  # 球拍長寬比門檻

# ===================== 手臂動作偵測參數 =====================
ARM_RAISE_THRESHOLD = 40   # 手腕高於肩膀的最小像素數
ARM_ELBOW_TOLERANCE = 20   # 手肘位置容錯範圍
ARM_COOLDOWN_FRAMES = 10   # 手臂抬起後的冷卻時間（幀）

# ===================== 擊球追蹤參數 =====================
WAIT_FRAMES = 5           # 手臂抬起後等待幀數
TRACK_FRAMES = 50         # 追蹤幀數
TOTAL_TRACKING_FRAMES = WAIT_FRAMES + TRACK_FRAMES  # 總追蹤幀數 (55)

# ===================== 擊球分類參數 =====================
# Smash/Drop 判斷門檻
SLOPE_THRESHOLD = 80       # 明顯下降的斜率門檻
SMASH_VELOCITY_THRESHOLD = 550  # Smash 速度門檻
SMASH_ACCELERATION_THRESHOLD = 2  # Smash 加速度門檻

# Clear 判斷門檻
CLEAR_SLOPE_THRESHOLD = -30  # 向上球的斜率門檻
CLEAR_HIGH_POSITION_RATIO = 0.3  # 最高點位置比例

# 其他判斷參數
HIGH_BALL_Y_THRESHOLD = 350  # 判斷為「高球」的 y 座標門檻
LOW_BALL_Y_THRESHOLD = 400   # 判斷為「低球」的 y 座標門檻

# ===================== 顯示設定 =====================
SHOT_DISPLAY_FRAMES = 75   # 擊球結果顯示持續時間（幀）
PROGRESS_UPDATE_INTERVAL = 10  # 進度更新間隔（秒）

# ===================== COCO 17-keypoint 骨架連線定義 =====================
KPT_PAIRS = [
    (5, 7), (7, 9),      # 左手：肩-肘-腕
    (6, 8), (8, 10),     # 右手
    (11, 13), (13, 15),  # 左腳：髖-膝-踝
    (12, 14), (14, 16),  # 右腳
    (5, 6),              # 雙肩
    (11, 12),            # 雙髖
    (5, 11), (6, 12),    # 身體兩側
]

# ===================== GPU 設定 =====================
def get_device():
    """自動偵測可用的運算裝置"""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
