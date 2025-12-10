# 🏸 羽球分析系統 - 完整指南

本文檔提供羽球分析系統的完整使用說明。

> **最後更新**: 2025-12-09  
> **狀態**: ✅ 系統已優化完成

---

## 📋 快速開始

### 1. 執行主程式
```bash
cd yufu
python testPlayerPoseEst.py
```

### 2. 操作說明
程式運行時會：
- 自動偵測人物和羽球
- 追蹤手臂動作
- 分析擊球類型

當偵測到擊球時會暫停，按鍵標註正確答案：
- **C** = Clear（高遠球）
- **S** = Smash（殺球）
- **D** = Drop（切球）
- **Space** = 接受 AI 判斷

其他控制：
- **P** = 暫停/繼續
- **Q** = 退出

---

## ⚙️ 配置說明

所有參數集中在 `config.py`，方便調整：

### 常用參數

```python
# 影片路徑
VIDEO_PATH = "./20250711_short.mp4"

# 羽球偵測信心度（越低越敏感）
SHUTTLECOCK_CONF = 0.15

# 擊球分類門檻
SLOPE_THRESHOLD = 80              # 下降斜率門檻
SMASH_VELOCITY_THRESHOLD = 550    # Smash 速度門檻
SMASH_ACCELERATION_THRESHOLD = 2  # Smash 加速度門檻
```

### 調整建議

**提高球偵測率**:
```python
SHUTTLECOCK_CONF = 0.10  # 降低門檻
BALL_DIFF_THRESH = 30    # 提高敏感度
```

**調整擊球判斷**:
```python
SLOPE_THRESHOLD = 100              # 更嚴格的下降判斷
SMASH_VELOCITY_THRESHOLD = 600     # Smash 需要更快
SMASH_ACCELERATION_THRESHOLD = 3   # 需要更明顯的加速
```

---

## 📊 系統架構

### 1. 人物偵測
- 使用 YOLOv8-Pose 模型
- 自動選擇場內球員（根據腳踝位置）
- 繪製骨架連線

### 2. 羽球偵測
- **主要方法**: YOLO 自訓練模型
  - 模型: `runs/detect/shuttlecock_improved_*/weights/best.pt`
  - 信心度門檻: 0.15
- **備用方法**: 傳統差分偵測
  - 當 YOLO 失效時自動切換

### 3. 手臂動作識別
- 偵測右手臂抬起（手腕高於肩膀 40px）
- 冷卻機制避免重複觸發（10 幀）
- 觸發後追蹤球運動 55 幀（5 幀等待 + 50 幀追蹤）

### 4. 擊球分類

使用多個參數綜合判斷：

| 參數 | 說明 |
|------|------|
| overall_slope | 球的整體 y 軸變化（正=下降，負=上升）|
| velocity | 平均速度 (px/s) |
| acceleration | 加速度變化（Smash 特徵）|
| highest_position_ratio | 最高點出現位置（0-1）|
| last_frames_low | 最後 5 幀是否在低處 |

**分類邏輯**:
```
如果 last_frames_low = True:
    如果 slope > 80 且 (acceleration > 2 或 velocity > 550):
        → Smash
    否則:
        → Drop
否則如果 slope < -30:
    → Clear
否則如果 highest_position_ratio < 0.3:
    → Clear
...
```

### 5. AI 輔助建議
- 基於歷史標註學習
- 相似度匹配找最接近的案例
- 當歷史數據 ≥ 5 筆時啟用

---

## 📁 檔案說明

### 核心檔案
- `testPlayerPoseEst.py` - 主程式
- `config.py` - 配置檔
- `shot_trajectories.json` - 軌跡標註記錄

### 資料檔案
- `court_pts.npy` - 球場邊界座標
- `ball_boundary.npy` - 球的有效範圍
- `20250711_short.mp4` - 測試影片

### 模型檔案
- `yolov8n-pose.pt` - 人物姿態模型
- `runs/detect/shuttlecock_improved_*/weights/best.pt` - 羽球偵測模型

---

## 🎯 標註建議

### 為什麼要標註？
1. **改進 AI 判斷**: 累積更多數據讓 AI 學習
2. **調整參數**: 分析錯誤案例，找出最佳門檻值
3. **建立訓練集**: 未來可訓練機器學習模型

### 標註目標
- **當前**: 14 筆
- **建議**: 50+ 筆
- **理想**: 100+ 筆

### 標註技巧
1. 觀察完整軌跡再判斷
2. 注意球的加速度變化（Smash 關鍵）
3. 檢查最後 5 幀位置（排除對方回擊）
4. 參考顯示的參數輔助判斷

---

## 🔧 故障排除

### 問題 1: 球偵測不到
**解決方案**:
```python
# config.py
SHUTTLECOCK_CONF = 0.10  # 降低門檻
SHOW_ALL_DETECTIONS = True  # 顯示所有候選
```

### 問題 2: 手臂動作誤觸發
**解決方案**:
```python
# config.py
ARM_RAISE_THRESHOLD = 50  # 提高門檻
ARM_COOLDOWN_FRAMES = 15  # 延長冷卻時間
```

### 問題 3: 擊球分類錯誤
**解決方案**:
1. 檢查 `shot_trajectories.json` 找出錯誤模式
2. 調整 `config.py` 中對應的門檻值
3. 累積更多標註數據

### 問題 4: 程式運行緩慢
**解決方案**:
- 確認使用 GPU（MPS/CUDA）
- 降低影片解析度
- 關閉 `SHOW_ALL_DETECTIONS`

---

## 📈 效能指標

### 處理速度
- **GPU (M1/M2)**: ~30-40 fps
- **CPU**: ~8-12 fps

### 準確度（當前）
- 球偵測: ~85%
- 擊球分類: ~60%（需更多標註改進）

---

## 🚀 進階功能

### 訓練新的羽球偵測模型

如需重新訓練模型（例如換不同場景的影片）：

1. **準備資料集**:
```bash
python improve_shuttlecock_detection.py --mode extract --num-frames 400
python improve_shuttlecock_detection.py --mode merge
```

2. **標註圖片**:
```bash
python simple_annotator.py train
python simple_annotator.py val
```

3. **訓練模型**:
```bash
python train_shuttlecock_detector.py --epochs 100
```

4. **更新配置**:
```python
# config.py
SHUTTLECOCK_WEIGHTS = "./runs/detect/新模型路徑/weights/best.pt"
```

---

## 📞 支援

遇到問題？
1. 檢查 [故障排除](#故障排除) 章節
2. 查看 `nohup.out` 日誌
3. 確認所有檔案路徑正確

---

**版本**: 2.0  
**日期**: 2025-12-09  
**狀態**: ✅ Production Ready
