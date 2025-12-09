# 🏸 羽球偵測改進指南

## 問題
目前的 YOLO 模型還有很多羽球沒有偵測到，需要增加訓練資料來改善。

## 解決方案：三步驟改進流程

### Step 1: 擷取未偵測到的羽球畫面
```bash
cd /Users/eric/Documents/UW\ 修課/2025\ fall/EEP\ 596A\ CV/EEP-596-Computer-Vision/final-project/yufu

python improve_shuttlecock_detection.py --mode extract --num-frames 50
```

**這個步驟會：**
- 自動分析影片，找出「有運動但沒偵測到球」的幀
- 選出品質最好的 50 張圖片
- 儲存到 `badminton_ball_dataset/images/additional/`

**預期輸出：**
```
🔍 Step 1: 尋找未被偵測到的羽球畫面...
📹 影片資訊：2258 幀, 60 FPS
✅ 找到 156 個候選幀
📸 選擇品質最好的 50 張
💾 儲存圖片到 badminton_ball_dataset/images/additional/...
✨ 完成！請使用 LabelImg 標註這些圖片
```

---

### Step 2: 標註新圖片
```bash
# 安裝 LabelImg（如果還沒裝）
pip install labelImg

# 開啟標註工具
labelImg badminton_ball_dataset/images/additional
```

**標註步驟：**
1. 打開 LabelImg
2. 點選 "Open Dir" → 選擇 `badminton_ball_dataset/images/additional`
3. 點選 "Change Save Dir" → 選擇相同的 `badminton_ball_dataset/images/additional`
4. 確認格式是 **YOLO** (左下角會顯示)
5. 使用 'W' 鍵框選羽球
6. 類別選擇 `shuttlecock` (class 0)
7. 按 'D' 下一張，繼續標註

**標註重點：**
- ✅ 即使羽球很小、模糊也要標註
- ✅ 羽球部分被遮擋也要標註可見的部分
- ✅ 盡量貼緊羽球邊緣框選
- ❌ 不要框太大（包含太多背景）

**鍵盤快捷鍵：**
- `W` - 建立框框
- `D` - 下一張圖片
- `A` - 上一張圖片
- `Ctrl+S` - 儲存

---

### Step 3: 重新訓練模型
```bash
python improve_shuttlecock_detection.py --mode train --epochs 100
```

**這個步驟會：**
1. 自動將新標註的圖片整合到現有資料集 (80% train, 20% val)
2. 使用擴充後的資料集重新訓練
3. 訓練 100 個 epochs（約需 10-20 分鐘）
4. 儲存最佳模型到 `runs/detect/shuttlecock_improved_*/weights/best.pt`

**訓練參數：**
- 使用 MPS (Apple Silicon GPU) 加速
- 針對小物體優化
- 加強資料增強（旋轉、翻轉、馬賽克等）

**預期輸出：**
```
🎯 Step 3: 重新訓練模型...
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100      2.1G      1.234      0.567      0.890        156        640
  ...
100/100      2.1G      0.345      0.123      0.234        156        640

✅ 訓練完成！
最佳模型：runs/detect/shuttlecock_improved_20241209_153045/weights/best.pt
```

---

### Step 4: 使用新模型
```bash
# 複製最佳模型
cp runs/detect/shuttlecock_improved_*/weights/best.pt ./shuttlecock_best_v2.pt

# 更新 testPlayerPoseEst.py
# 修改第 28 行：
# SHUTTLECOCK_WEIGHTS = "./shuttlecock_best_v2.pt"

# 重新執行分析
python testPlayerPoseEst.py
```

---

## 快速一鍵流程（進階）
如果你想一次完成所有步驟（中間會暫停讓你標註）：

```bash
python improve_shuttlecock_detection.py --mode full --num-frames 50 --epochs 100
```

---

## 調整參數

### 擷取更多/更少圖片
```bash
python improve_shuttlecock_detection.py --mode extract --num-frames 100  # 擷取 100 張
```

### 訓練更久/更短
```bash
python improve_shuttlecock_detection.py --mode train --epochs 150  # 訓練 150 epochs
```

### 使用不同影片
```bash
python improve_shuttlecock_detection.py --mode extract --video ./other_video.mp4
```

---

## 預期改善效果

**改善前：**
- 偵測率：~60%
- 很多小球、遠距離球沒偵測到

**改善後（增加 50+ 標註）：**
- 偵測率：~80-85%
- 小球偵測改善
- 遠距離球偵測改善
- 模糊球偵測改善

**如果還不夠好：**
- 再重複 Step 1-3，再增加 50-100 張標註
- 或標註其他影片的羽球
- 目標：總共 200-300 張標註圖片

---

## 疑難排解

### Q: 找不到模型檔案
```bash
# 檢查模型路徑
ls -la runs/detect/shuttlecock_train/weights/best.pt

# 如果找不到，指定正確路徑
python improve_shuttlecock_detection.py --mode extract --model <你的模型路徑>
```

### Q: LabelImg 無法開啟
```bash
# 重新安裝
pip uninstall labelImg
pip install labelImg

# 或使用其他標註工具（如 CVAT, Roboflow）
```

### Q: 訓練太慢
```bash
# 減少 epochs
python improve_shuttlecock_detection.py --mode train --epochs 50

# 減少 batch size（如果記憶體不足）
python improve_shuttlecock_detection.py --mode train --batch 8
```

### Q: 想看訓練進度
訓練時會自動儲存到 `runs/detect/shuttlecock_improved_*/`，裡面有：
- `results.png` - 訓練曲線圖
- `confusion_matrix.png` - 混淆矩陣
- `val_batch*.jpg` - 驗證集預測結果

---

## 建議標註策略

**第一輪（50 張）：**
- 專注在「現有模型完全漏掉」的情況
- 小球、模糊球、遮擋球

**第二輪（50 張）：**
- 不同角度、不同光線
- 不同背景位置

**第三輪（100 張）：**
- 其他影片的羽球
- 確保資料多樣性

**目標：總共 200-300 張高品質標註**

---

## 下一步

1. ✅ 執行 `--mode extract` 擷取圖片
2. ✅ 使用 LabelImg 標註
3. ✅ 執行 `--mode train` 訓練
4. ✅ 測試新模型效果
5. 🔄 如果效果還不夠好，重複 1-4

Good luck! 🚀
