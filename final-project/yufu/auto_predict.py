#!/usr/bin/env python3
"""
自動預測羽球影片分析
完全自動化處理，不需要任何人工標註或互動
只需提供影片路徑，系統會自動：
1. 偵測羽球位置
2. 追蹤軌跡
3. 判斷擊球類型（Clear/Smash/Drop）
4. 輸出帶有標註的影片
"""

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
import os

# Fix for PyTorch 2.6+ weights_only loading issue
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.PoseModel'])

class AutoBadmintonAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        
        # 自動設定輸出路徑
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path) or "."
        self.output_path = os.path.join(output_dir, f"{base_name}_analyzed.mp4")
        
        # 模型路徑（使用相對路徑）
        script_dir = os.path.dirname(__file__)
        self.pose_weights = os.path.join(script_dir, "yolov8n-pose.pt")
        self.shuttlecock_weights = os.path.join(script_dir, "runs/detect/shuttlecock_improved_20251209_122742/weights/best.pt")
        
        # 球場邊界（自動計算或使用預設）
        self.court_pts_path = os.path.join(script_dir, "court_pts.npy")
        
        # YOLO 羽球偵測參數
        self.shuttlecock_conf = 0.15
        self.show_all_detections = True
        
        # 球的運動限制參數
        self.max_ball_speed = 2000
        self.max_ball_y = 650
        self.min_ball_y = 30
        
        # 追蹤參數
        self.max_tracking_frames = 50
        self.min_tracking_frames = 15
        self.shot_display_duration = 90
        
        # COCO 17-keypoint skeleton 連線定義
        self.kpt_pairs = [
            (5, 7), (7, 9),      # 左手
            (6, 8), (8, 10),     # 右手
            (11, 13), (13, 15),  # 左腳
            (12, 14), (14, 16),  # 右腳
            (5, 6),              # 雙肩
            (11, 12),            # 雙髖
            (5, 11), (6, 12),    # 身體兩側
        ]
        
        print(f"📹 輸入影片: {self.video_path}")
        print(f"📤 輸出影片: {self.output_path}")
        
    def load_models(self):
        """載入 YOLO 模型"""
        print("\n🔄 載入 AI 模型...")
        
        # 載入姿態估計模型
        if not os.path.exists(self.pose_weights):
            print(f"⚠️  找不到姿態模型: {self.pose_weights}")
            print("請確認 yolov8n-pose.pt 存在")
            sys.exit(1)
        self.pose_model = YOLO(self.pose_weights)
        print(f"✅ 姿態模型載入完成")
        
        # 載入羽球偵測模型
        if not os.path.exists(self.shuttlecock_weights):
            print(f"⚠️  找不到羽球偵測模型: {self.shuttlecock_weights}")
            print("將使用基礎偵測方法")
            self.shuttlecock_model = None
        else:
            self.shuttlecock_model = YOLO(self.shuttlecock_weights)
            print(f"✅ 羽球偵測模型載入完成")
    
    def load_court_boundary(self, frame_width, frame_height):
        """載入或計算球場邊界"""
        if os.path.exists(self.court_pts_path):
            court_pts = np.load(self.court_pts_path, allow_pickle=True)
            print(f"✅ 載入球場邊界座標")
            return court_pts
        else:
            print(f"⚠️  未找到球場座標，使用預設邊界")
            # 預設使用整個畫面作為球場區域
            return np.array([
                [0, 0],
                [frame_width, 0],
                [frame_width, frame_height],
                [0, frame_height]
            ])
    
    def detect_shuttlecock_yolo(self, frame):
        """使用 YOLO 偵測羽球"""
        if self.shuttlecock_model is None:
            return None, []
        
        results = self.shuttlecock_model(frame, conf=self.shuttlecock_conf, verbose=False)
        
        all_detections = []
        best_detection = None
        best_conf = 0
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # 計算中心點
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                detection_info = {
                    'bbox': (int(x1), int(y1), int(w), int(h)),
                    'center': (int(cx), int(cy)),
                    'conf': conf,
                    'filtered': False,
                    'reason': ''
                }
                
                # 基本過濾
                if cy > self.max_ball_y:
                    detection_info['filtered'] = True
                    detection_info['reason'] = 'y too low'
                elif cy < self.min_ball_y:
                    detection_info['filtered'] = True
                    detection_info['reason'] = 'y too high'
                elif conf > best_conf:
                    best_conf = conf
                    best_detection = detection_info
                
                all_detections.append(detection_info)
        
        return best_detection, all_detections
    
    def classify_shot(self, tracking_points, frame_idx):
        """根據軌跡分類擊球類型"""
        if len(tracking_points) < self.min_tracking_frames:
            return "Unknown", {}
        
        # 計算基本軌跡特徵
        first_y = tracking_points[0][1]
        last_y = tracking_points[-1][1]
        head_to_tail_dy = last_y - first_y
        
        min_y = min(pt[1] for pt in tracking_points)
        max_y = max(pt[1] for pt in tracking_points)
        y_range = max_y - min_y
        
        # 計算速度
        total_distance = 0
        for i in range(1, len(tracking_points)):
            dx = tracking_points[i][0] - tracking_points[i-1][0]
            dy = tracking_points[i][1] - tracking_points[i-1][1]
            total_distance += np.hypot(dx, dy)
        velocity = total_distance / len(tracking_points) * 30  # 假設 30 fps
        
        # 計算高處停留比例
        mid_y = (min_y + max_y) / 2
        high_ball_count = sum(1 for pt in tracking_points if pt[1] < mid_y)
        high_ball_ratio = high_ball_count / len(tracking_points)
        
        # 檢查最後幾幀是否在低處
        last_frames_low = sum(1 for pt in tracking_points[-5:] if pt[1] > 400) >= 3
        
        # 計算加速度
        acceleration = 0
        if len(tracking_points) >= 20:
            mid_idx = len(tracking_points) // 2
            first_half_dist = sum(
                np.hypot(tracking_points[i][0] - tracking_points[i-1][0],
                        tracking_points[i][1] - tracking_points[i-1][1])
                for i in range(1, mid_idx)
            )
            second_half_dist = sum(
                np.hypot(tracking_points[i][0] - tracking_points[i-1][0],
                        tracking_points[i][1] - tracking_points[i-1][1])
                for i in range(mid_idx, len(tracking_points))
            )
            first_half_speed = first_half_dist / mid_idx if mid_idx > 0 else 0
            second_half_speed = second_half_dist / (len(tracking_points) - mid_idx)
            acceleration = second_half_speed - first_half_speed
        
        # 分類邏輯
        shot_type = "Unknown"
        
        if last_frames_low:
            # 結束在低處 → 排除 Clear
            if head_to_tail_dy > 80:
                if acceleration > 2 or velocity > 550:
                    shot_type = "Smash"
                else:
                    shot_type = "Drop"
            else:
                shot_type = "Drop"
        else:
            # 結束在高處 → 可能是 Clear
            if head_to_tail_dy < -50 or high_ball_ratio > 0.6:
                shot_type = "Clear"
            elif head_to_tail_dy > 80:
                if acceleration > 2 or velocity > 550:
                    shot_type = "Smash"
                else:
                    shot_type = "Drop"
            else:
                shot_type = "Unknown"
        
        # 返回分類結果和軌跡資訊
        trajectory_info = {
            'velocity': velocity,
            'dy': head_to_tail_dy,
            'high_ball_ratio': high_ball_ratio,
            'y_range': y_range,
            'acceleration': acceleration
        }
        
        return shot_type, trajectory_info
    
    def draw_skeleton(self, frame, keypoints):
        """繪製人體骨架"""
        for (i, j) in self.kpt_pairs:
            if i < len(keypoints) and j < len(keypoints):
                pt1 = keypoints[i]
                pt2 = keypoints[j]
                if pt1[2] > 0.3 and pt2[2] > 0.3:  # 置信度門檻
                    cv2.line(frame, 
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            (0, 255, 0), 2)
    
    def process_video(self):
        """主要處理流程"""
        # 開啟影片
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ 無法開啟影片: {self.video_path}")
            return
        
        # 取得影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📊 影片資訊:")
        print(f"   解析度: {w}x{h}")
        print(f"   FPS: {fps:.2f}")
        print(f"   總幀數: {total_frames}")
        print(f"   時長: {total_frames/fps:.2f} 秒")
        
        # 載入模型
        self.load_models()
        
        # 載入球場邊界
        court_pts = self.load_court_boundary(w, h)
        x_coords = [pt[0] for pt in court_pts]
        x_min_ball = min(x_coords)
        x_max_ball = max(x_coords)
        
        # 設定輸出影片
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        
        # 初始化追蹤變數
        tracking_points = []
        last_ball_pos = None
        shot_type = "Unknown"
        shot_display_timer = 0
        trajectory_info = {}
        frame_idx = 0
        
        # 進度顯示
        start_time = time.time()
        last_progress_time = start_time
        
        print(f"\n🚀 開始處理影片...")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # === 1. 姿態估計 ===
            pose_results = self.pose_model(frame, verbose=False)
            
            # 繪製骨架
            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                kpts_data = pose_results[0].keypoints.data.cpu().numpy()
                for person_kpts in kpts_data:
                    self.draw_skeleton(frame, person_kpts)
            
            # === 2. 羽球偵測 ===
            ball_detection, all_detections = self.detect_shuttlecock_yolo(frame)
            
            current_ball = None
            if ball_detection and not ball_detection['filtered']:
                cx, cy = ball_detection['center']
                # 檢查是否在球場範圍內
                if x_min_ball <= cx <= x_max_ball:
                    current_ball = (cx, cy)
                    
                    # 繪製球的位置
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)
            
            # === 3. 軌跡追蹤 ===
            if current_ball:
                tracking_points.append(current_ball)
                
                # 繪製軌跡
                if len(tracking_points) >= 2:
                    for i in range(1, len(tracking_points)):
                        cv2.line(frame, tracking_points[i-1], tracking_points[i],
                                (255, 0, 255), 2)
                
                # 達到最大追蹤幀數 → 判斷擊球類型
                if len(tracking_points) >= self.max_tracking_frames:
                    shot_type, trajectory_info = self.classify_shot(tracking_points, frame_idx)
                    shot_display_timer = self.shot_display_duration
                    
                    print(f"🎯 Frame {frame_idx}: 偵測到擊球 → {shot_type}")
                    
                    # 重置追蹤
                    tracking_points = []
            
            elif len(tracking_points) > 0:
                # 球消失了 → 如果有足夠的軌跡，進行分類
                if len(tracking_points) >= self.min_tracking_frames:
                    shot_type, trajectory_info = self.classify_shot(tracking_points, frame_idx)
                    shot_display_timer = self.shot_display_duration
                    
                    print(f"🎯 Frame {frame_idx}: 偵測到擊球 → {shot_type}")
                
                tracking_points = []
            
            # === 4. 繪製資訊面板 ===
            # 幀數顯示（右上角）
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", 
                       (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 擊球結果顯示（畫面上方中央）
            if shot_display_timer > 0:
                if trajectory_info:
                    panel_width = 500
                    panel_x = (w - panel_width) // 2
                    panel_y = 60
                    
                    shot_info = [
                        f"SHOT: {shot_type}",
                        f"Velocity: {trajectory_info.get('velocity', 0):.1f} px/s",
                        f"Delta-Y: {trajectory_info.get('dy', 0):.1f} px",
                        f"High Ratio: {trajectory_info.get('high_ball_ratio', 0):.1%}"
                    ]
                    
                    line_height = 30
                    panel_height = 20 + len(shot_info) * line_height
                    
                    # 半透明背景
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (panel_x, panel_y),
                                (panel_x + panel_width, panel_y + panel_height),
                                (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                    
                    # 顯示文字
                    for idx, text in enumerate(shot_info):
                        y_pos = panel_y + 25 + idx * line_height
                        if idx == 0:
                            cv2.putText(frame, text, (panel_x + 15, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                        else:
                            cv2.putText(frame, text, (panel_x + 15, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                
                shot_display_timer -= 1
            
            # === 5. 輸出影片 ===
            out.write(frame)
            
            # 即時顯示（可選）
            cv2.imshow('Auto Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  使用者中斷處理")
                break
            
            frame_idx += 1
            
            # 進度顯示（每10秒）
            current_time = time.time()
            if current_time - last_progress_time >= 10.0:
                elapsed = current_time - start_time
                progress = (frame_idx / total_frames) * 100
                fps_process = frame_idx / elapsed
                eta = (total_frames - frame_idx) / fps_process if fps_process > 0 else 0
                
                print(f"進度: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                      f"速度: {fps_process:.1f} fps | "
                      f"預估剩餘: {eta:.1f}秒")
                
                last_progress_time = current_time
        
        # 釋放資源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # 最終統計
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        
        print("\n" + "=" * 60)
        print("✅ 處理完成！")
        print("=" * 60)
        print(f"總幀數: {frame_idx}")
        print(f"總耗時: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
        print(f"平均處理速度: {avg_fps:.1f} fps")
        print(f"輸出檔案: {self.output_path}")
        print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("使用方法: python auto_predict.py <影片路徑>")
        print("範例: python auto_predict.py ./my_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ 找不到影片: {video_path}")
        sys.exit(1)
    
    analyzer = AutoBadmintonAnalyzer(video_path)
    analyzer.process_video()


if __name__ == "__main__":
    main()
