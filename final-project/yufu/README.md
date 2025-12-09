# 羽球分析系統 - 專案說明

## 📁 專案結構

```
yufu/
├── config.py                    # ⭐ 統一配置檔（所有參數集中管理）
├── testPlayerPoseEst.py        # 主程式（完整分析系統）
├── shot_annotations.json       # 擊球標註記錄
├── court_pts.npy               # 球場邊界座標
├── ball_boundary.npy           # 球的有效範圍
├── 20250711_short.mp4          # 測試影片
├── output_with_pose.mp4        # 輸出影片
│
├── archive/                    # 舊版本檔案
├── result/                     # 輸出結果
├── runs/                       # YOLO 訓練結果
└── badminton_ball_dataset/     # 羽球訓練資料集
```

## 🚀 快速開始

### 1. 執行主程式
```bash
cd yufu
python testPlayerPoseEst.py
```

### 2. 修改參數
編輯 `config.py` 調整所有設定：
- 影片路徑
- 偵測門檻
- 擊球分類參數

### 3. 標註擊球類型
程式會在偵測到擊球時暫停，按鍵標註：
- `C` = Clear（高遠球）
- `S` = Smash（殺球）  
- `D` = Drop（切球）
- `Space` = 接受 AI 判斷

## 📊 主要功能

1. **人物偵測與骨架追蹤**
   - 使用 YOLOv8-Pose
   - 自動選擇場內球員

2. **羽球偵測**
   - YOLO 自訓練模型
   - 傳統差分法備用

3. **擊球動作識別**
   - 手臂抬起偵測
   - 球軌跡追蹤（55 幀）
   - 自動分類：Smash/Drop/Clear

4. **AI 輔助判斷**
   - 基於歷史標註學習
   - 提供判斷建議

## 🔧 常用調整

### 提高球偵測準確度
```python
# config.py
SHUTTLECOCK_CONF = 0.10  # 降低門檻（更多候選）
BALL_DIFF_THRESH = 30    # 提高敏感度
```

### 調整擊球分類門檻
```python
# config.py
SLOPE_THRESHOLD = 100              # 提高 = 更嚴格判斷為下降球
SMASH_VELOCITY_THRESHOLD = 600     # 提高 = Smash 需要更快
SMASH_ACCELERATION_THRESHOLD = 3   # 提高 = Smash 需要更明顯加速
```

## 📝 標註記錄

所有標註自動儲存在 `shot_annotations.json`，包含：
- 時間戳
- 幀數
- 軌跡參數（斜率、速度、加速度等）
- AI 預測結果
- 用戶標註
- 正確性

## ⚠️ 注意事項

1. **首次執行**需要：
   - 標註球場邊界（`mark_court.py`）
   - 標註球的範圍（`mark_ball_boundary.py`）

2. **模型檔案**需下載：
   - `yolov8n-pose.pt`（人物姿態）
   - 羽球模型在 `runs/detect/shuttlecock_improved_*/weights/best.pt`

3. **GPU 加速**：
   - 自動偵測 CUDA/MPS
   - M1/M2 Mac 使用 MPS 可達 30+ fps

## 📈 改進建議

持續標註擊球類型以提升 AI 準確度！
目前標註數：14 筆（建議 > 50 筆）
