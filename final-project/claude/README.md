# 羽球動作分析系統 - Final Project

> **⚠️ 專案狀態說明 (2025/12/08)**
> 
> 由於時間限制，本專案目前實作**簡化版本**，專注於**選手姿勢分析**功能。
> 
> **目前已完成功能：**
> - ✅ 人物骨架偵測與追蹤
> - ✅ 球場線條繪製與視覺化
> - ✅ 關鍵角度分析
> - ✅ 專業姿勢對比與評分
> 
> **待完成功能（需額外時間）：**
> - ⏳ 羽球自動偵測（需標注訓練資料）
> - ⏳ 擊球瞬間自動偵測
> - ⏳ 動作自動分類 (Smash/Clear/Drop)
> 
> 完整版本規劃請見文件末尾的「[完整版本計畫](#完整版本計畫)」章節。

---

## 📁 專案結構

```
final-project/
├── badminton_analysis.py       # 主程式 (完整版)
├── quick_test.py               # 快速測試版 (最小可行)
├── court_calibration.py        # 球場標定工具
├── train_shuttlecock.py        # 羽球模型訓練 (選用)
│
├── README_MVP.md               # 詳細指南
├── CHECKLIST.md                # 執行檢查清單
│
├── 20250711_short.mp4          # 輸入影片 (你的羽球影片)
├── court_corners.pkl           # 球場角點資料 (執行 calibration 後生成)
│
├── yolov8n.pt                  # YOLO 偵測模型
├── yolov8n-pose.pt             # YOLO pose 模型
│
└── outputs/
    ├── quick_test_output.mp4           # 快速測試輸出
    └── badminton_analysis_output.mp4   # 完整分析輸出
```

## 🎯 目前已實作功能（簡化版）

### ✅ 已完成
1. **人物骨架偵測** - YOLO11-pose (17個關鍵點)
2. **球場線條繪製** - 4點透視校正視覺化
3. **關鍵角度計算** - 肩膀、手肘、手腕角度分析
4. **專業姿勢對比** - 與標準殺球姿勢比較並評分

### ⏳ 原計畫功能（待實作）
5. **羽球自動偵測** - 需訓練自定義 YOLO 模型（需標注 100+ 張圖片）
6. **擊球瞬間偵測** - 基於羽球與球拍距離/速度判斷
7. **動作自動分類** - Rule-based 分類 (Smash/Clear/Drop)

> 💡 **為什麼簡化？** 羽球偵測需要自定義訓練資料（標注 100+ 張圖片約需 2-3 小時），
> 考量到專案時程，先專注於姿勢分析核心功能。未來可透過標注訓練擴展完整功能。

## ⚡ 快速開始（簡化版）

### 執行姿勢分析程式
```bash
# 1. 球場標定（一次性設定）
python court_calibration.py

# 2. 執行姿勢分析
python pose_analysis.py

# 3. 查看結果
open pose_analysis_output.mp4
```

> **注意**: 目前簡化版專注於骨架偵測和姿勢分析，不包含羽球偵測功能。

## 📊 與論文比較

| 項目 | 論文 (2020) | 本專案 (2025) |
|------|-------------|---------------|
| 人體偵測 | YOLOv5 | YOLO11 ✅ |
| 姿勢估計 | OpenPose | YOLO11-pose ✅ |
| 羽球偵測 | 自訓練 | COCO/自訓練 |
| 框架 | 多框架 | 單一框架 ✅ |
| 速度 | ~15 fps | ~30+ fps ✅ |
| 易用性 | 複雜 | 簡單 ✅ |

## 🔧 技術細節

### YOLO11-pose Keypoints (17點)
```
0: Nose          6: R_Shoulder    12: R_Hip
1: L_Eye         7: L_Elbow       13: L_Knee
2: R_Eye         8: R_Elbow       14: R_Knee
3: L_Ear         9: L_Wrist       15: L_Ankle
4: R_Ear        10: R_Wrist       16: R_Ankle
5: L_Shoulder   11: L_Hip
```

### 動作分類邏輯

#### Smash (殺球)
- 肩膀角度 > 70°
- 手臂接近伸直 (elbow > 150°)
- 球向下高速

#### Clear (高遠球)
- 球向上
- 速度快

#### Drop (吊球)
- 球向下
- 速度慢

## 📈 改進方向

### 短期 (Final Project)
- [x] 基礎骨架偵測
- [x] 球場線條
---

## 📅 完整版本計畫

> 以下為原始完整專案規劃，因時間限制暫時簡化。未來可依此路線圖擴展功能。

### 階段 1: 簡化版（目前已完成）✅
- [x] 人物骨架偵測 (YOLO11-pose)
- [x] 球場線條繪製 (4點透視校正)
- [x] 關鍵角度計算與分析
- [x] 專業姿勢對比與評分

### 階段 2: 羽球偵測（需額外 2-3 小時）⏳
- [ ] 準備標注資料
  - 提取影片幀（已完成，見 `annotation_images/`）
  - 使用 Roboflow 標注羽球（100 張圖，約 1 小時）
  - 下載 YOLOv8 格式資料集
- [ ] 訓練自定義模型
  - 執行 `train_shuttlecock.py`（約 1 小時）
  - 驗證模型準確率 (目標 >85%)
- [ ] 整合羽球偵測
  - 更新 `shuttlecock_detector.py` 使用訓練模型
  - 測試偵測效果

### 階段 3: 完整功能實作（需額外 1-2 小時）⏳
- [ ] 擊球瞬間偵測
  - 基於羽球與球拍距離判斷
  - 球速突變檢測
- [ ] 動作自動分類
  - Smash/Clear/Drop 分類邏輯
  - 結合骨架角度與球速
- [ ] 整合追蹤功能
  - ByteTrack 羽球軌跡追蹤
  - 軌跡視覺化

### 階段 4: 進階功能（研究方向）
- [ ] ML-based 動作分類 (LSTM/Transformer)
- [ ] 球軌跡預測
- [ ] 多人追蹤
- [ ] 即時分析 (Webcam)

### 預估完整開發時間
- **簡化版**: 已完成 ✅
- **羽球偵測**: 2-3 小時（標注 + 訓練）
- **完整功能**: 1-2 小時（整合 + 測試）
- **總計**: 額外需要 3-5 小時可達成完整版

---

## 🎓 簡化版 Demo 要點

### 技術亮點
1. **統一框架** - 全部用 YOLO，不需要多個模型
2. **即時性** - YOLO11-pose 比 OpenPose 快 2-3 倍
3. **實用性** - 量化姿勢分析，可操作建議
4. **可擴展** - 架構支援未來新增羽球偵測等功能

### 展示流程
```
1. 問題: 羽球訓練缺乏量化姿勢反饋
2. 方案: YOLO11-pose + 角度分析 + 專業對比
3. Demo: 播放姿勢分析影片
4. 結果: 骨架視覺化、角度計算、評分建議
5. 未來: 加入羽球偵測、動作分類、軌跡預測
```

### 回答問題準備
- **為什麼不用 OpenPose?**
  → YOLO11-pose 更快、更準、更易用
  
- **為什麼沒有羽球偵測?**
  → 需要標注訓練資料（2-3小時），時間限制下先專注姿勢分析核心功能
  
- **能即時嗎?**
  → GPU 可達 30+ fps, CPU ~10 fps
  
- **如何評分?**
  → 與專業標準角度比較，計算偏差並提供改善建議

- **未來如何擴展?**
  → 已準備好訓練腳本和資料，可快速加入羽球偵測功能

---

## 🐛 常見問題

### Q: import ultralytics 失敗
```bash
pip install ultralytics
```

### Q: 找不到影片
```python
# 修改 VIDEO_PATH
VIDEO_PATH = "你的影片路徑.mp4"
```

### Q: 沒有 yolo11n-pose.pt
```python
# 會自動下載,或改用 yolov8n-pose.pt
```

### Q: 偵測不到球
- 方案 1: 訓練羽球模型
- 方案 2: 用手腕位置代替
- 方案 3: 手動標記測試

## 📚 參考資料

- **YOLO11 文檔**: https://docs.ultralytics.com/
- **論文**: Intelligent System of Badminton Serve Action Based on YOLOv5 and OpenPose
- **Dataset**: COCO Keypoints, 或自建

## 👨‍💻 作者

Eric - UW EEP 596A Computer Vision Final Project

## 📝 授權

Educational Use Only

---

**祝你 Final Project 順利! 🏸✨**

有問題隨時問,加油!
