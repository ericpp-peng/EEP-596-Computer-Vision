# 🏸 羽球分析系統 - Badminton Analysis System

完整的羽球影片分析系統，包含球員偵測、羽球追蹤、球場區域判斷與球種分類。

---

## 📚 文檔導航（重要！）

**本專案包含以下文檔，請依需求閱讀：**

### 🎯 球種分類系統（Shot Classification - AI）
1. **[docs/QUICK_START.md](./docs/QUICK_START.md)** ⭐ **從這裡開始！**
   - 球種分類系統快速入門
   - 3 步驟快速使用
   - 常見問題解答
   
2. **[docs/SHOT_CLASSIFIER_README.md](./docs/SHOT_CLASSIFIER_README.md)**
   - 完整的球種分類系統說明
   - 模型架構與訓練流程
   - 進階功能與參數調整
   
3. **[docs/TRAINING_COMPLETE_REPORT.md](./docs/TRAINING_COMPLETE_REPORT.md)**
   - 球種分類訓練結果報告
   - 模型效能評估
   - 改進建議

### 🏸 羽球偵測系統（Shuttlecock Detection - YOLO）
4. **[docs/SHUTTLECOCK_TRAINING_GUIDE.md](./docs/SHUTTLECOCK_TRAINING_GUIDE.md)**
   - YOLO 羽球偵測模型訓練指南
   - 資料集準備與標註
   - 訓練參數優化

---

## 📁 專案結構

```
yufu/
├── 📄 README.md                       # ⭐ 本檔案（專案總覽 + 文檔導航）
│
├── 📂 docs/                           # 📚 所有詳細文檔都在這裡
│   ├── QUICK_START.md                 # 球種分類快速入門
│   ├── SHOT_CLASSIFIER_README.md      # 球種分類完整說明
│   ├── TRAINING_COMPLETE_REPORT.md    # 球種分類訓練報告
│   └── SHUTTLECOCK_TRAINING_GUIDE.md  # 羽球偵測訓練指南
│
├── 🎯 主程式
│   └── testPlayerPoseEst.py          # 主分析程式
│
├── 🤖 球種分類 AI 系統（核心功能）
│   ├── shot_classifier_model.py      # 神經網路模型定義
│   ├── shot_dataset.py               # 資料集處理與載入
│   ├── train_shot_classifier.py      # 訓練腳本
│   ├── shot_classifier_inference.py  # 推理模組（用於整合）
│   ├── integration_example.py        # 整合到主程式的範例代碼
│   └── verify_system.py              # 系統驗證與測試工具
│
├── 🔧 標註與準備工具
│   ├── mark_court.py                 # 球場邊界標註工具
│   ├── mark_ball_boundary.py         # 球的有效範圍標註工具
│   ├── simple_annotator.py           # 簡易標註工具
│   └── extract_frames_for_annotation.py  # 影格提取工具
│
├── 📊 資料與模型檔案
│   ├── shot_annotations.json         # 球種標註資料（22 樣本）
│   ├── court_pts.npy                 # 球場座標
│   ├── ball_boundary.npy             # 球界座標
│   ├── shot_classifier_weights/      # AI 模型權重目錄
│   │   ├── best_model_*.pth          # 最佳模型權重
│   │   ├── scaler.pkl                # 特徵標準化參數
│   │   ├── confusion_matrix.png      # 混淆矩陣
│   │   └── training_history.png      # 訓練曲線
│   └── runs/                          # YOLO 訓練結果
│
└── 🎓 YOLO 模型
    ├── yolov8n-pose.pt               # YOLO 姿態估計模型
    └── yolov8n.pt                    # YOLO 物件偵測模型
```

## 🚀 快速開始

### 1️⃣ 驗證球種分類系統

```bash
python verify_system.py
```

### 2️⃣ 執行主分析程式

```bash
python testPlayerPoseEst.py
# 按 1/2/3 標註球種（Smash/Clear/Drop）
```

### 3️⃣ 訓練或重新訓練球種分類模型

```bash
python train_shot_classifier.py
```

---

## 🎯 系統功能

### ✅ 已實現
- **球員偵測**：YOLOv8-Pose，17 個關鍵點追蹤
- **羽球追蹤**：YOLO 自訓練模型 + 卡爾曼濾波
- **球場區域判斷**：左/右場區分
- **擊球偵測**：手腕-球距離 + 速度變化
- **球種分類 AI**：Smash（殺球）/ Clear（高遠球）/ Drop（切球）
  - 規則判斷（基於軌跡參數）
  - AI 神經網路分類器（已訓練，60% 準確率）

### 📊 目前狀態（2025-12-09）
```
✅ 系統功能完整
✅ AI 模型已訓練
📊 標註資料：22 樣本
🎯 測試準確率：60%
⚠️  建議：收集 50-100 個樣本以提升準確率
```

---

## 💡 使用指南（給其他人看）

### 🆕 第一次使用這個專案？

**推薦流程：**

1. **閱讀本 README** - 了解專案概況（就是現在這個檔案）
2. **閱讀 [docs/QUICK_START.md](./docs/QUICK_START.md)** - 快速上手球種分類系統
3. **執行驗證**：`python verify_system.py`
4. **開始使用**：`python testPlayerPoseEst.py`

### 🔧 想要改進或訓練模型？

**深入了解：**

1. **球種分類系統** → [docs/SHOT_CLASSIFIER_README.md](./docs/SHOT_CLASSIFIER_README.md)
   - 完整的模型架構說明
   - 訓練流程與參數調整
   - 整合到主程式的方法

2. **訓練結果分析** → [docs/TRAINING_COMPLETE_REPORT.md](./docs/TRAINING_COMPLETE_REPORT.md)
   - 目前模型的效能評估
   - 錯誤分析與改進建議

3. **羽球偵測訓練** → [docs/SHUTTLECOCK_TRAINING_GUIDE.md](./docs/SHUTTLECOCK_TRAINING_GUIDE.md)
   - YOLO 模型訓練指南
   - 資料集準備方法

4. **整合範例** → [integration_example.py](./integration_example.py)
   - 如何將 AI 整合到主程式的代碼範例

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
1. ✨ **收集更多標註資料**（最重要！目標 50+ 樣本）
2. 🔧 將 AI 分類器整合到主程式
3. 📊 重新訓練提高準確率

### 中長期
1. 🎥 支援更多影片格式
2. 📈 即時分析優化
3. 🤖 新增更多球種（網前球、平抽等）
4. 📊 比賽統計分析功能

---

**需要幫助？查看 `docs/` 目錄中的詳細文檔！** 📚

**Have fun! 🏸🚀**
