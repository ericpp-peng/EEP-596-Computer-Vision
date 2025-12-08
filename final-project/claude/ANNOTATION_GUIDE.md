# 羽球標注指南

## 📦 已完成
- ✅ 已提取 99 張圖片到 `annotation_images/` 資料夾

## 🎯 標注方法（三選一）

### 方法 1: Roboflow（最推薦，免費且簡單）

1. **註冊帳號**: https://roboflow.com
2. **建立專案**: 
   - Project Name: `Badminton Shuttlecock`
   - Project Type: `Object Detection`
   - Annotation Group: `shuttlecock`

3. **上傳圖片**:
   - 將 `annotation_images/` 資料夾中的所有圖片上傳

4. **開始標注**:
   - 使用矩形框標記每個羽球
   - 標籤名稱: `shuttlecock`
   - 技巧: 羽球通常是白色小圓形物體

5. **匯出資料集**:
   - Format: `YOLOv8`
   - Split: Train 70%, Valid 20%, Test 10%
   - 下載 zip 檔案

6. **解壓縮**:
   ```bash
   unzip Badminton-Shuttlecock.zip -d badminton_dataset
   ```

---

### 方法 2: LabelImg（本地工具）

1. **安裝**:
   ```bash
   pip install labelImg
   ```

2. **啟動**:
   ```bash
   labelImg annotation_images/
   ```

3. **標注步驟**:
   - 按 `W` 畫矩形框
   - 輸入類別: `shuttlecock`
   - 按 `D` 下一張圖片
   - 儲存格式選擇: `YOLO`

4. **準備資料集結構**:
   ```
   badminton_dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

5. **建立 `badminton.yaml`**:
   ```yaml
   path: ./badminton_dataset
   train: images/train
   val: images/val
   
   names:
     0: shuttlecock
   ```

---

### 方法 3: CVAT（進階，團隊協作）

1. **註冊**: https://cvat.ai
2. **建立任務**: Object Detection
3. **上傳圖片**: 批次上傳所有圖片
4. **標注**: 使用矩形工具標記羽球
5. **匯出**: YOLOv5 1.1 格式（相容 YOLOv8）

---

## 🏷️ 標注技巧

### 羽球特徵：
- **顏色**: 白色（通常是畫面中最亮的小物體）
- **形狀**: 小圓形或橢圓形
- **運動**: 快速移動
- **模糊**: 可能有運動模糊

### 標注原則：
1. **框要緊**: 矩形框盡量貼近羽球
2. **包含模糊**: 如果有運動模糊，也要包含在框內
3. **小心誤標**: 不要標記燈光、地板反光等
4. **遮擋處理**: 即使部分被遮擋，也要標記可見部分

### 標注示例：
```
正確 ✅:
┌─┐  <- 緊貼羽球的小框
│●│
└─┘

錯誤 ❌:
┌───────┐  <- 框太大
│   ●   │
└───────┘
```

---

## 📊 建議標注數量

- **最少**: 50 張（快速測試）
- **推薦**: 100 張（已提取）
- **理想**: 200+ 張（高準確率）

---

## 🚀 標注完成後

### 1. 檢查資料集結構
```
badminton_dataset/
├── data.yaml (或 badminton.yaml)
├── images/
│   ├── train/  (約 70 張)
│   └── val/    (約 30 張)
└── labels/
    ├── train/  (約 70 個 .txt)
    └── val/    (約 30 個 .txt)
```

### 2. 修改 `train_shuttlecock.py`
確認 `DATA_YAML` 路徑正確：
```python
DATA_YAML = "badminton_dataset/data.yaml"
```

### 3. 開始訓練
```bash
python train_shuttlecock.py
```

### 4. 訓練完成後
模型會儲存在:
```
runs/detect/badminton/weights/best.pt
```

### 5. 更新偵測器使用自定義模型
修改 `shuttlecock_detector.py`:
```python
SHUTTLECOCK_MODE = "custom"
SHUTTLECOCK_MODEL = "runs/detect/badminton/weights/best.pt"
```

---

## ❓ 常見問題

**Q: 羽球太小看不清楚？**
A: 可以放大圖片檢視，標注時框要小而精準

**Q: 羽球模糊怎麼辦？**
A: 仍要標注，包含模糊區域，這能幫助模型學習運動中的羽球

**Q: 同一幀有多顆羽球？**
A: 正常比賽只有一顆，標注看起來最像真球的那顆

**Q: 沒看到羽球？**
A: 可以跳過該圖片，或標記場地上的球

**Q: 標注要多久？**
A: 使用 Roboflow 約 30-60 分鐘可完成 100 張

---

## 📞 需要協助？

如果標注過程有問題，可以：
1. 先標注 20-30 張測試訓練效果
2. 使用 Roboflow 的自動標注輔助功能
3. 考慮使用預訓練的小球偵測模型微調

---

**準備好了嗎？選擇一個工具開始標注吧！** 🎯
