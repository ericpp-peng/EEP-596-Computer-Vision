"""
球場標定工具
用滑鼠點擊4個角點來標定球場位置
使用方式:
    python court_calibration.py
    
點擊順序: 左上 -> 右上 -> 右下 -> 左下
按 's' 儲存, 按 'q' 退出
"""

import cv2
import numpy as np
import pickle

VIDEO_PATH = "20250711_short.mp4"
CORNERS_FILE = "court_corners.pkl"

# 全域變數
corners = []
current_frame = None


def mouse_callback(event, x, y, flags, param):
    """滑鼠點擊事件"""
    global corners, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append([x, y])
            print(f"點 {len(corners)}: ({x}, {y})")
            
            # 重繪
            display_frame = current_frame.copy()
            for i, pt in enumerate(corners):
                cv2.circle(display_frame, tuple(pt), 8, (0, 255, 0), -1)
                cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 如果有至少2點, 畫線
            if len(corners) > 1:
                for i in range(len(corners)-1):
                    cv2.line(display_frame, tuple(corners[i]), 
                            tuple(corners[i+1]), (255, 100, 0), 2)
            
            # 如果4點都標了, 畫完整框
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(display_frame, [pts], isClosed=True, 
                             color=(255, 100, 0), thickness=2)
                
                # 畫中線
                mid_top = ((pts[0] + pts[1]) // 2).tolist()
                mid_bot = ((pts[3] + pts[2]) // 2).tolist()
                cv2.line(display_frame, mid_top, mid_bot, (255, 100, 0), 2)
            
            cv2.imshow("Court Calibration", display_frame)


def main():
    global corners, current_frame
    
    # 讀取影片第一幀
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"無法開啟影片: {VIDEO_PATH}")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("無法讀取影片第一幀")
        return
    
    current_frame = frame.copy()
    
    # 建立視窗
    cv2.namedWindow("Court Calibration")
    cv2.setMouseCallback("Court Calibration", mouse_callback)
    
    print("=" * 60)
    print("球場標定工具")
    print("=" * 60)
    print("請依序點擊球場的 4 個角點:")
    print("  1. 左上角")
    print("  2. 右上角")
    print("  3. 右下角")
    print("  4. 左下角")
    print("\n按 's' 儲存, 按 'r' 重設, 按 'q' 退出")
    print("=" * 60)
    
    cv2.imshow("Court Calibration", current_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("退出")
            break
        
        elif key == ord('r'):
            print("重設角點")
            corners = []
            cv2.imshow("Court Calibration", current_frame)
        
        elif key == ord('s'):
            if len(corners) == 4:
                # 儲存角點
                with open(CORNERS_FILE, 'wb') as f:
                    pickle.dump(corners, f)
                print(f"\n✅ 角點已儲存到 {CORNERS_FILE}")
                print(f"角點座標: {corners}")
                break
            else:
                print(f"⚠️  請先標記 4 個角點 (目前: {len(corners)}/4)")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
