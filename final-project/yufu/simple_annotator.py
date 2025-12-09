#!/usr/bin/env python3
"""
簡易羽球標註工具
解決 LabelImg 在 macOS Python 3.13 的兼容性問題

使用方法：
    python simple_annotator.py

操作說明：
    - 滑鼠左鍵拖曳：框選羽球
    - 空白鍵：儲存並下一張
    - D 鍵：不標註，直接下一張
    - A 鍵：回上一張
    - Q 鍵：退出
    - R 鍵：重新框選（清除當前標註）
"""

import cv2
import os
import glob
from pathlib import Path

class SimpleAnnotator:
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        
        # 取得所有圖片
        self.image_files = sorted(glob.glob(str(self.image_dir / "*.jpg")))
        if not self.image_files:
            raise ValueError(f"在 {image_dir} 找不到任何 .jpg 檔案")
        
        self.current_idx = 0
        self.boxes = []  # 當前圖片的標註框
        self.drawing = False
        self.start_point = None
        self.current_point = None
        
        print(f"🎯 找到 {len(self.image_files)} 張圖片")
        print(f"📁 標註檔將儲存在：{self.label_dir}")
        print("\n" + "=" * 60)
        print("操作說明：")
        print("  滑鼠左鍵拖曳：框選羽球")
        print("  空白鍵：儲存並下一張")
        print("  D 鍵：不標註，直接下一張")
        print("  A 鍵：回上一張")
        print("  R 鍵：重新框選（清除當前標註）")
        print("  Q 鍵：退出")
        print("=" * 60 + "\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠事件處理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # 確保座標正確（左上到右下）
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 檢查框是否太小
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.boxes.append((x1, y1, x2, y2))
                    print(f"  ✓ 新增標註框：({x1}, {y1}) -> ({x2}, {y2})")
                
                self.start_point = None
                self.current_point = None
    
    def draw_boxes(self, img):
        """繪製標註框"""
        display_img = img.copy()
        
        # 繪製已儲存的框
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_img, f"Ball {i+1}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 繪製正在拖曳的框
        if self.drawing and self.start_point and self.current_point:
            x1, y1 = self.start_point
            x2, y2 = self.current_point
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        return display_img
    
    def save_annotation(self, img_path):
        """儲存 YOLO 格式標註"""
        if not self.boxes:
            return
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 產生對應的標註檔名
        img_name = Path(img_path).stem
        label_path = self.label_dir / f"{img_name}.txt"
        
        with open(label_path, 'w') as f:
            for (x1, y1, x2, y2) in self.boxes:
                # 轉換為 YOLO 格式（歸一化）
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # YOLO 格式：class x_center y_center width height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"  💾 已儲存標註：{label_path}")
    
    def load_annotation(self, img_path):
        """載入已有的標註（如果存在）"""
        img_name = Path(img_path).stem
        label_path = self.label_dir / f"{img_name}.txt"
        
        self.boxes = []
        
        if label_path.exists():
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)
                        
                        # 轉換回像素座標
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        self.boxes.append((x1, y1, x2, y2))
            
            if self.boxes:
                print(f"  📂 載入已有標註：{len(self.boxes)} 個框")
    
    def run(self):
        """執行標註"""
        cv2.namedWindow('Badminton Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Badminton Annotator', self.mouse_callback)
        
        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"❌ 無法讀取圖片：{img_path}")
                self.current_idx += 1
                continue
            
            # 載入已有標註
            self.load_annotation(img_path)
            
            print(f"\n📷 [{self.current_idx + 1}/{len(self.image_files)}] {Path(img_path).name}")
            if self.boxes:
                print(f"   已有 {len(self.boxes)} 個標註框")
            
            while True:
                display_img = self.draw_boxes(img)
                
                # 顯示進度和提示
                cv2.putText(display_img, 
                           f"Image {self.current_idx + 1}/{len(self.image_files)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(display_img, 
                           f"Boxes: {len(self.boxes)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(display_img, 
                           "SPACE=Save & Next | D=Skip | A=Prev | R=Reset | Q=Quit", 
                           (10, display_img.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Badminton Annotator', display_img)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # 空白鍵：儲存並下一張
                    self.save_annotation(img_path)
                    self.current_idx += 1
                    break
                
                elif key == ord('d') or key == ord('D'):  # D：跳過
                    print("  ⏭️  跳過此圖")
                    self.current_idx += 1
                    break
                
                elif key == ord('a') or key == ord('A'):  # A：上一張
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        print("  ⏮️  回到上一張")
                    break
                
                elif key == ord('r') or key == ord('R'):  # R：重新標註
                    self.boxes = []
                    print("  🔄 清除標註")
                
                elif key == ord('q') or key == ord('Q'):  # Q：退出
                    print("\n👋 標註已結束")
                    cv2.destroyAllWindows()
                    return
        
        print("\n" + "=" * 60)
        print("🎉 所有圖片標註完成！")
        print("=" * 60)
        cv2.destroyAllWindows()


def main():
    import sys
    
    # 檢查參數
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        # 預設標註 train
        mode = 'train'
    
    if mode == 'train':
        image_dir = "badminton_ball_dataset/images/train"
        label_dir = "badminton_ball_dataset/labels/train"
        print("\n🏋️ 標註 Train 資料集")
    elif mode == 'val':
        image_dir = "badminton_ball_dataset/images/val"
        label_dir = "badminton_ball_dataset/labels/val"
        print("\n✅ 標註 Val 資料集")
    else:
        print("❌ 未知模式。使用方法：")
        print("   python simple_annotator.py train   # 標註訓練集")
        print("   python simple_annotator.py val     # 標註驗證集")
        return
    
    try:
        annotator = SimpleAnnotator(image_dir, label_dir)
        annotator.run()
        
        # 統計標註完成度
        label_files = list(Path(label_dir).glob("*.txt"))
        image_files = list(Path(image_dir).glob("*.jpg"))
        
        print(f"\n📊 標註統計：")
        print(f"   圖片總數：{len(image_files)}")
        print(f"   已標註：{len(label_files)}")
        print(f"   未標註：{len(image_files) - len(label_files)}")
        
        if mode == 'train' and len(label_files) == len(image_files):
            print(f"\n✅ Train 標註完成！")
            print(f"下一步：標註 Val 資料集")
            print(f"指令：python simple_annotator.py val")
        elif mode == 'val' and len(label_files) == len(image_files):
            print(f"\n🎉 所有標註完成！可以開始訓練了！")
            print(f"指令：python train_shuttlecock_detector.py")
    
    except Exception as e:
        print(f"❌ 錯誤：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
