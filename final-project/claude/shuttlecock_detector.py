"""
羽球偵測器 - 進階版
結合多種策略提高羽球偵測準確率:
1. YOLO 預訓練模型 (sports ball)
2. 基於顏色的偵測 (白色羽球)
3. 動態追蹤 (卡爾曼濾波)
4. 手腕位置估計 (備用方案)
"""

import cv2
import numpy as np
import os
from collections import deque

# 修正 PyTorch 2.6 weights_only 問題
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO

class ShuttlecockDetector:
    """
    羽球偵測器
    支援多種偵測模式
    """
    
    def __init__(self, mode='hybrid', model_path='yolov8n.pt', custom_model_path=None, court_area=None):
        """
        mode: 'yolo', 'color', 'hybrid', 'custom'
        - yolo: 使用 COCO sports ball
        - color: 基於白色偵測
        - hybrid: 結合 YOLO + color
        - custom: 使用自訓練模型
        
        court_area: 場地範圍限制 (可選)
        - None: 不限制
        - 'auto': 自動偵測 (使用畫面中心區域)
        - (x, y, w, h): 矩形範圍
        - [(x1,y1), (x2,y2), ...]: 多邊形頂點
        """
        self.mode = mode
        self.model = None
        self.history = deque(maxlen=30)
        self.tracker = None
        self.court_area = court_area
        self.court_polygon = None
        
        # 載入模型
        if mode in ['yolo', 'hybrid']:
            self.model = YOLO(model_path)
            print(f"✅ 載入 YOLO 模型: {model_path}")
        
        if mode == 'custom' and custom_model_path:
            self.model = YOLO(custom_model_path)
            print(f"✅ 載入自訓練模型: {custom_model_path}")
        
        # 設置場地範圍
        if court_area is not None and court_area != 'auto':
            if isinstance(court_area, (list, tuple)) and len(court_area) == 4:
                # 檢查是否為矩形 (x, y, w, h)
                if all(isinstance(x, (int, float)) for x in court_area):
                    x, y, w, h = court_area
                    self.court_polygon = np.array([
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ], dtype=np.int32)
                else:
                    # 多邊形頂點
                    self.court_polygon = np.array(court_area, dtype=np.int32)
            print(f"✅ 場地範圍限制已啟用")
    
    
    def set_auto_court_area(self, frame):
        """
        自動設置場地範圍 (使用畫面中心 85% 區域)
        """
        h, w = frame.shape[:2]
        margin_x = int(w * 0.075)  # 左右各留 7.5%
        margin_y = int(h * 0.075)  # 上下各留 7.5%
        
        self.court_polygon = np.array([
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y]
        ], dtype=np.int32)
        print(f"✅ 自動設置場地範圍: {margin_x},{margin_y} -> {w-margin_x},{h-margin_y}")
    
    
    def is_in_court(self, point):
        """
        檢查點是否在場地範圍內
        """
        if self.court_polygon is None:
            return True  # 沒有限制,全部接受
        
        x, y = int(point[0]), int(point[1])
        result = cv2.pointPolygonTest(self.court_polygon, (x, y), False)
        return result >= 0  # >= 0 表示在多邊形內或邊界上
    
    
    def detect_by_yolo(self, frame):
        """
        使用 YOLO 偵測球
        COCO class 32 = sports ball (包括網球、籃球等)
        """
        if self.model is None:
            return None
        
        results = self.model(frame, imgsz=640)[0]
        
        balls = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Sports ball 或自訓練的羽球類別
            if cls_id == 32 or cls_id == 0:  # 0 通常是自訓練的第一個類別
                if conf > 0.3:  # 降低信心閾值
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # 檢查是否在場地範圍內
                    if not self.is_in_court(center):
                        continue  # 跳過場地外的偵測
                    
                    size = max(x2 - x1, y2 - y1)
                    balls.append({
                        'pos': center,
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'size': size
                    })
        
        # 選擇最可能的球 (最小的高信心物體)
        if balls:
            # 羽球通常很小
            balls.sort(key=lambda x: x['size'])
            return balls[0]
        
        return None
    
    
    def detect_by_color(self, frame):
        """
        基於顏色偵測白色羽球
        適用於背景較暗的場景
        """
        # 轉換到 HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 白色範圍 (較寬鬆)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        
        # 二值化
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 形態學處理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 羽球面積範圍 (根據實際情況調整)
            if 10 < area < 500:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # 接近圓形或橢圓
                if 0.5 < aspect_ratio < 2.0:
                    center = (x + w//2, y + h//2)
                    
                    # 檢查是否在場地範圍內
                    if not self.is_in_court(center):
                        continue  # 跳過場地外的偵測
                    
                    candidates.append({
                        'pos': center,
                        'bbox': (x, y, x+w, y+h),
                        'area': area,
                        'conf': min(area / 200, 1.0)  # 面積轉信心度
                    })
        
        # 選最大的
        if candidates:
            candidates.sort(key=lambda x: x['area'], reverse=True)
            return candidates[0]
        
        return None
    
    
    def detect_by_wrist(self, keypoints):
        """
        使用手腕位置估計球的位置 (備用方案)
        假設球在右手腕附近
        """
        if keypoints is None or len(keypoints) < 11:
            return None
        
        # 右手腕 = index 10
        r_wrist = keypoints[10]
        
        if r_wrist[0] > 0 and r_wrist[1] > 0:
            return {
                'pos': tuple(r_wrist),
                'bbox': (r_wrist[0]-10, r_wrist[1]-10, r_wrist[0]+10, r_wrist[1]+10),
                'conf': 0.5,
                'source': 'wrist'
            }
        
        return None
    
    
    def detect(self, frame, keypoints=None):
        """
        主偵測函式
        根據 mode 選擇策略
        """
        # 自動設置場地範圍 (只在第一次執行)
        if self.court_area == 'auto' and self.court_polygon is None:
            self.set_auto_court_area(frame)
        
        result = None
        
        if self.mode == 'yolo':
            result = self.detect_by_yolo(frame)
        
        elif self.mode == 'color':
            result = self.detect_by_color(frame)
        
        elif self.mode == 'hybrid':
            # 先試 YOLO
            result = self.detect_by_yolo(frame)
            
            # 如果信心度不高,試顏色偵測
            if result is None or result['conf'] < 0.5:
                color_result = self.detect_by_color(frame)
                if color_result and color_result['conf'] > 0.6:
                    result = color_result
        
        elif self.mode == 'custom':
            result = self.detect_by_yolo(frame)
        
        # 如果都失敗,用手腕估計
        if result is None and keypoints is not None:
            result = self.detect_by_wrist(keypoints)
        
        # 加入歷史
        if result:
            self.history.append(result['pos'])
        
        return result
    
    
    def get_velocity(self):
        """
        計算球速向量
        """
        if len(self.history) < 2:
            return np.array([0, 0])
        
        return np.array(self.history[-1]) - np.array(self.history[-2])
    
    
    def draw(self, frame, detection, show_court=True):
        """
        視覺化偵測結果
        """
        # 繪製場地範圍 (半透明)
        if show_court and self.court_polygon is not None:
            overlay = frame.copy()
            cv2.polylines(overlay, [self.court_polygon], True, (0, 255, 0), 2)
            cv2.fillPoly(overlay, [self.court_polygon], (0, 255, 0))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        if detection is None:
            return frame
        
        x, y = detection['pos']
        x, y = int(x), int(y)
        
        # 畫圓點
        cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
        cv2.circle(frame, (x, y), 12, (255, 255, 0), 2)
        
        # 畫 bbox
        if 'bbox' in detection:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # 顯示信心度
        conf = detection.get('conf', 0)
        source = detection.get('source', 'detect')
        cv2.putText(frame, f"Ball ({source}): {conf:.2f}", 
                   (x + 15, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 畫軌跡
        if len(self.history) > 1:
            points = np.array(self.history, dtype=np.int32)
            cv2.polylines(frame, [points], False, (255, 200, 0), 2)
        
        return frame


class RacketDetector:
    """
    球拍偵測器
    """
    
    def __init__(self, mode='wrist'):
        """
        mode: 'yolo', 'wrist'
        - yolo: 使用 COCO tennis racket (class 38)
        - wrist: 直接用手腕位置
        """
        self.mode = mode
    
    
    def detect(self, frame, keypoints=None, model=None):
        """
        偵測球拍位置
        """
        if self.mode == 'yolo' and model:
            results = model(frame, imgsz=640)[0]
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 38:  # tennis racket
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    return {
                        'pos': center,
                        'bbox': (x1, y1, x2, y2)
                    }
        
        # 備用: 用手腕
        if keypoints is not None and len(keypoints) > 10:
            r_wrist = keypoints[10]
            if r_wrist[0] > 0 and r_wrist[1] > 0:
                return {
                    'pos': tuple(r_wrist),
                    'source': 'wrist'
                }
        
        return None


# ===== 測試程式 =====
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("羽球偵測器測試")
    print("=" * 60)
    
    # 測試不同模式
    modes = ['yolo', 'color', 'hybrid']
    
    video_path = "20250711_short.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        sys.exit(1)
    
    # 測試 hybrid 模式
    detector = ShuttlecockDetector(mode='hybrid')
    
    print("\n按 'q' 退出, 'space' 暫停")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 偵測
        result = detector.detect(frame)
        
        # 視覺化
        detector.draw(frame, result)
        
        # 顯示
        cv2.imshow("Shuttlecock Detection Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ 測試完成")
