"""
互動式羽球邊界線標註工具
使用方法：
1. 執行程式: python mark_ball_boundary.py
2. 用滑鼠點擊左邊界線上的任意一點
3. 用滑鼠點擊右邊界線上的任意一點
4. 程式會自動畫出兩條垂直線
5. 按 's' 儲存
6. 按 'r' 重新開始
7. 按 'q' 退出
"""

import cv2
import numpy as np

# ==================== 設定 ====================
VIDEO_PATH = "./Chou-pro.mp4"     # 影片路徑
OUTPUT_PATH = "./ball_boundary_pro.npy"     # 輸出的邊界線坐標檔案
COURT_PTS_PATH = "./court_pts_pro.npy"      # 球場坐標（可選，用來顯示球場）
# ===============================================

class BallBoundaryMarker:
    def __init__(self, video_path, output_path, court_pts_path=None):
        self.video_path = video_path
        self.output_path = output_path
        self.court_pts_path = court_pts_path
        self.points = []  # 存 2 個點: [左邊界點, 右邊界點]
        self.frame = None
        self.display_frame = None
        self.scale = 1.0
        self.court_pts = None
        self.frame_height = 0
        
        # 載入球場座標（如果有的話）
        if court_pts_path:
            try:
                self.court_pts = np.load(court_pts_path)
                print(f"✅ 已載入球場座標: {court_pts_path}")
            except:
                print(f"⚠️  無法載入球場座標: {court_pts_path}")
        
    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠點擊事件處理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                # 將顯示座標轉換回原始座標
                original_x = int(x / self.scale)
                original_y = int(y / self.scale)
                self.points.append([original_x, original_y])
                
                point_names = ["左邊界", "右邊界"]
                print(f"點 {len(self.points)} ({point_names[len(self.points)-1]}): 顯示({x}, {y}) -> 原始 x={original_x}")
                self.update_display()
    
    def update_display(self):
        """更新顯示畫面"""
        self.display_frame = self.frame.copy()
        h = self.display_frame.shape[0]
        
        # 畫球場（如果有的話）
        if self.court_pts is not None:
            court_contour = self.court_pts.reshape((-1, 1, 2)).astype(np.int32)
            display_court = (court_contour * self.scale).astype(np.int32)
            cv2.polylines(self.display_frame, [display_court], 
                         isClosed=True, color=(255, 0, 0), thickness=2)
        
        # 畫垂直邊界線
        point_names = ["左邊界", "右邊界"]
        
        for i, pt in enumerate(self.points):
            display_x = int(pt[0] * self.scale)
            display_y = int(pt[1] * self.scale)
            
            # 畫垂直線（從頂到底）
            cv2.line(self.display_frame, (display_x, 0), (display_x, h-1), (0, 255, 255), 2)
            
            # 畫點擊的點
            cv2.circle(self.display_frame, (display_x, display_y), 8, (0, 255, 255), -1)
            
            # 標註文字
            cv2.putText(self.display_frame, point_names[i], 
                       (display_x+15, display_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 顯示提示
        if len(self.points) >= 2:
            cv2.putText(self.display_frame, "Press 's' to save, 'r' to reset", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            remaining = 2 - len(self.points)
            next_point = point_names[len(self.points)] if len(self.points) < 2 else ""
            cv2.putText(self.display_frame, f"Click {remaining} more point(s) - Next: {next_point}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Ball Boundary Marker", self.display_frame)
    
    def reset(self):
        """重置標註"""
        self.points = []
        self.update_display()
        print("已重置，請重新標註")
    
    def save(self):
        """儲存邊界線座標"""
        if len(self.points) < 2:
            print(f"⚠️  需要標註 2 個點，目前只有 {len(self.points)} 個")
            return False
        
        # 直接使用兩個點的 x 座標
        left_x = int(self.points[0][0])
        right_x = int(self.points[1][0])
        
        boundary_data = {
            'points': np.array(self.points, dtype=np.int32),
            'left_x': left_x,
            'right_x': right_x
        }
        
        np.save(self.output_path, boundary_data)
        print(f"✅ 已儲存羽球邊界線到: {self.output_path}")
        print(f"標註點:\n{np.array(self.points)}")
        print(f"左邊界 x = {left_x}, 右邊界 x = {right_x}")
        return True
    
    def run(self):
        """執行標註工具"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ 無法開啟影片: {self.video_path}")
            return
        
        ret, original_frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ 無法讀取影片幀")
            return
        
        # 儲存原始尺寸
        original_h, original_w = original_frame.shape[:2]
        self.frame_height = original_h
        print(f"原始影片尺寸: {original_w} x {original_h}")
        
        # 縮放畫面
        max_width = 1280
        if original_w > max_width:
            self.scale = max_width / original_w
            new_w = int(original_w * self.scale)
            new_h = int(original_h * self.scale)
            self.frame = cv2.resize(original_frame, (new_w, new_h))
            print(f"顯示尺寸: {new_w}x{new_h} (縮放比例: {self.scale:.3f})")
        else:
            self.scale = 1.0
            self.frame = original_frame
            print("使用原始尺寸")
        
        self.display_frame = self.frame.copy()
        
        cv2.namedWindow("Ball Boundary Marker")
        cv2.setMouseCallback("Ball Boundary Marker", self.mouse_callback)
        
        print("=" * 60)
        print("羽球邊界線標註工具")
        print("=" * 60)
        print("1. 依序點擊 2 個點：")
        print("   - 左邊界（點擊左邊界線上的任意位置）")
        print("   - 右邊界（點擊右邊界線上的任意位置）")
        print("   程式會自動畫出垂直線")
        print("2. 按 's' 儲存")
        print("3. 按 'r' 重新開始")
        print("4. 按 'q' 退出")
        print("=" * 60)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("退出")
                break
            elif key == ord('s'):
                if self.save():
                    break
            elif key == ord('r'):
                self.reset()
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    marker = BallBoundaryMarker(VIDEO_PATH, OUTPUT_PATH, COURT_PTS_PATH)
    marker.run()
