# 🏸 羽球分析系統 - Badminton Analysis System

完整的羽球影片分析系統，包含球員偵測、羽球追蹤、球場區域判斷與球種分類。

---

## 📚 文檔導航（重要！）

**本專案包含以下文檔，請依需求閱讀：**

### 🏸 羽球偵測系統（Shuttlecock Detection - YOLO）
- **[docs/SHUTTLECOCK_TRAINING_GUIDE.md](./docs/SHUTTLECOCK_TRAINING_GUIDE.md)**
  - YOLO 羽球偵測模型訓練指南
  - 資料集準備與標註
  - 訓練參數優化

---

## 📁 專案結構

```
yufu/
├── 📄 README.md                       # ⭐ 本檔案（專案總覽 + 文檔導航）
│
├── 📂 docs/                           # 📚 詳細文檔
│   └── SHUTTLECOCK_TRAINING_GUIDE.md  # 羽球偵測訓練指南
│
├── 🎯 主程式
│   └── testPlayerPoseEst.py          # 主分析程式
│
├── 🤖 軌跡記錄系統
│   └── shot_trajectory_logger.py     # 完整 x,y 軌跡記錄器
│
├── 🔧 標註與準備工具
│   ├── mark_court.py                 # 球場邊界標註工具
│   ├── mark_ball_boundary.py         # 球的有效範圍標註工具
│   ├── simple_annotator.py           # 簡易標註工具
│   └── extract_frames_for_annotation.py  # 影格提取工具
│
├── 📊 資料與模型檔案
│   ├── shot_trajectories.json        # ⭐ 球種標註資料（完整 x,y 軌跡）
│   ├── court_pts.npy                 # 球場座標
│   ├── ball_boundary.npy             # 球界座標
│   └── runs/                          # YOLO 訓練結果
│
└── 🎓 YOLO 模型
    ├── yolov8n-pose.pt               # YOLO 姿態估計模型
    └── yolov8n.pt                    # YOLO 物件偵測模型
```

## 🚀 快速開始

### 1️⃣ 執行主分析程式

```bash
python testPlayerPoseEst.py
# 按 1/2/3 標註球種（Smash/Clear/Drop）
```

### 2️⃣ 查看收集的軌跡資料

```bash
python -c "import json; data = json.load(open('shot_trajectories.json')); print(f'共 {len(data)} 筆')"
```

---

## 🎯 系統功能

### ✅ 已實現
- **球員偵測**：YOLOv8-Pose，17 個關鍵點追蹤
- **羽球追蹤**：YOLO 自訓練模型 + 卡爾曼濾波
- **球場區域判斷**：左/右場區分
- **擊球偵測**：手腕-球距離 + 速度變化
- **球種分類 AI**：Smash（殺球）/ Clear（高遠球）/ Drop（切球）
  - ⭐ **軌跡記錄系統**（完整 x,y 座標，供未來訓練深度學習模型）

### 📊 目前狀態（2025-12-09）
```
✅ 系統功能完整
✅ 新版軌跡記錄系統已上線
📊 資料格式：完整 45 幀 x,y 座標 + 缺幀標記
🎯 平均偵測率：78.5%
🚀 下一步：收集 50+ 樣本訓練 LSTM/Transformer 模型
```

### 🆕 資料格式升級（重要！）

**新格式** (`shot_trajectories.json`)：
```json
{
  "timestamp": "2025-12-09T15:48:09.925652",
  "start_frame": 108,
  "end_frame": 153,
  "trajectory": [
    {"frame": 108, "x": 320.5, "y": 180.2, "detected": true},
    {"frame": 109, "x": null, "y": null, "detected": false},
    {"frame": 110, "x": 325.1, "y": 185.3, "detected": true}
  ],
  "user_label": "Clear",
  "stats": {
    "total_frames": 45,
    "detected_frames": 35,
    "detection_rate": 0.78
  }
}
```

**優勢**：
- ✅ 保留完整原始資料
- ✅ 讓神經網路自己學習特徵
- ✅ 可視覺化完整軌跡
- ✅ 未來可用 LSTM/Transformer

---

## 💡 使用指南（給其他人看）

### 🆕 第一次使用這個專案？

**推薦流程：**

1. **閱讀本 README** - 了解專案概況（就是現在這個檔案）
2. **執行主程式**：`python testPlayerPoseEst.py`
3. **開始標註**：按 1/2/3 標註球種（資料會儲存到 `shot_trajectories.json`）

### 🔧 想要改進或訓練模型？

**深入了解：**

1. **軌跡資料格式** → 查看 `shot_trajectories.json`
   - 完整的 45 幀 x, y 座標
   - 缺幀標記與偵測率統計

2. **羽球偵測訓練** → [docs/SHUTTLECOCK_TRAINING_GUIDE.md](./docs/SHUTTLECOCK_TRAINING_GUIDE.md)
   - YOLO 模型訓練指南
   - 資料集準備方法

3. **軌跡記錄器** → `shot_trajectory_logger.py`
   - 如何記錄和載入軌跡資料
   - 提供給深度學習模型的資料格式

---

## 📦 依賴套件

```bash
# 使用 conda（推薦）
conda install pytorch opencv numpy scikit-learn matplotlib seaborn
conda install -c conda-forge ultralytics

# 或使用 pip
pip install ultralytics opencv-python numpy torch scikit-learn matplotlib seaborn
```

---

## 🎓 技術要點

本專案展示了：
- ✅ **完整 ML 流程**（資料收集→訓練→評估→部署）
- ✅ **小數據處理技巧**（輕量模型、Dropout、類別權重平衡）
- ✅ **YOLO 物件偵測**（人體姿態 + 羽球偵測）
- ✅ **卡爾曼濾波**（軌跡平滑）
- ✅ **模組化設計**（易於擴展與整合）

---

## 📝 下一步建議

### 短期（1-2 週）
1. ✨ **收集更多軌跡資料**（最重要！目標 50+ 樣本）
2. 🤖 **訓練 LSTM/Transformer 模型**（基於完整軌跡）
3. 📊 比較新舊模型效能
4. 🔧 整合最佳模型到主程式

### 中長期
1. 📈 即時軌跡視覺化
2. 🤖 新增更多球種（網前球、平抽等）
3. 📊 比賽統計分析功能
4. 🎥 支援更多影片格式

---

**需要幫助？查看 `docs/` 目錄中的詳細文檔！** 📚

**Have fun! 🏸🚀**
