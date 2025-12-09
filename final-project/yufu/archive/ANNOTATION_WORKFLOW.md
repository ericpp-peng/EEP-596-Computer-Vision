# 🏸 羽球標註工作流程

## ✅ 當前狀態
已擷取 **334 張圖片** 到：
```
badminton_ball_dataset/images/additional/
```

包含：
- 2 張未偵測到的幀（高運動分數）
- 332 張已偵測但清晰的幀（增加多樣性）

---

## 📝 標註步驟

### Step 1: 合併圖片到訓練集
將 additional 資料夾的圖片分配到 train/val：

```bash
cd "/Users/eric/Documents/UW 修課/2025 fall/EEP 596A CV/EEP-596-Computer-Vision/final-project/yufu"

python improve_shuttlecock_detection.py --mode merge
```

這會：
- 80% 圖片 → `badminton_ball_dataset/images/train/`
- 20% 圖片 → `badminton_ball_dataset/images/val/`
- 自動建立對應的 labels 目錄

### Step 2: 標註訓練集
使用 simple_annotator.py 標註：

```bash
python simple_annotator.py train
```

#### 操作說明
- **滑鼠左鍵拖曳** → 畫標註框（框住羽球）
- **SPACE** → 儲存並進入下一張 ✅
- **D** → 跳過此圖（沒有清楚的球）⏭️
- **A** → 回到上一張 ⏮️
- **R** → 重新標註（清除所有框）🔄
- **Q** → 結束標註 👋

#### 標註重點
✅ **要標註的球：**
- 清晰可見的球
- 球體完整（遮擋 < 50%）
- 大小 > 5x5 像素

❌ **不要標註的球：**
- 太模糊看不清
- 被嚴重遮擋（> 50%）
- 太小難以辨識

**框要緊貼球體**，不要留太多空白。

### Step 3: 標註驗證集
完成訓練集後，標註驗證集：

```bash
python simple_annotator.py val
```

操作方式完全相同。

### Step 4: 檢查標註完成度
```bash
# 檢查訓練集標註數量
ls badminton_ball_dataset/labels/train/*.txt | wc -l

# 檢查驗證集標註數量
ls badminton_ball_dataset/labels/val/*.txt | wc -l
```

---

## 🚀 訓練模型

標註完成後，訓練改進的模型：

```bash
python improve_shuttlecock_detection.py --mode train --epochs 100
```

訓練參數：
- Epochs: 100（可調整）
- Batch size: 16（可用 `--batch` 調整）
- 使用現有資料 + 新標註資料

---

## 📊 使用新模型

訓練完成後，模型在：
```
runs/detect/shuttlecock_improved_YYYYMMDD_HHMMSS/weights/best.pt
```

### 更新 testPlayerPoseEst.py
找到這行：
```python
SHUTTLECOCK_WEIGHTS = "./runs/detect/shuttlecock_train/weights/best.pt"
```

改成：
```python
SHUTTLECOCK_WEIGHTS = "./runs/detect/shuttlecock_improved_YYYYMMDD_HHMMSS/weights/best.pt"
```

---

## 📈 預期改進
新模型應該能：
- ✅ 偵測到更多之前漏掉的球
- ✅ 在不同光線/角度下更穩定
- ✅ 減少誤判（因為包含了更多樣的訓練資料）

---

## ⚠️ 注意事項

1. **標註品質 > 數量**
   - 寧可少標但準確，也不要多標但不精確

2. **框的大小**
   - 緊貼球體，不要框太大
   - 參考已有標註檔的格式

3. **進度保存**
   - simple_annotator.py 會自動保存標註
   - 可以隨時中斷（Q），下次繼續

4. **標註一致性**
   - train 和 val 用同樣標準
   - 所有球都要標（不要漏標清楚的球）

---

## 🔧 故障排除

### 問題：simple_annotator.py 無法執行
```bash
# 確保在正確目錄
cd "/Users/eric/Documents/UW 修課/2025 fall/EEP 596A CV/EEP-596-Computer-Vision/final-project/yufu"

# 確認 Python 環境
which python
```

### 問題：訓練失敗
檢查：
1. 是否標註了 train 和 val 兩個資料集
2. 標註檔案格式是否正確（YOLO 格式）
3. data.yaml 設定是否正確

### 問題：模型效果不佳
可能原因：
1. 標註品質不佳（框不準確）
2. 標註數量太少（建議至少 50 張）
3. 訓練 epochs 不夠（增加到 150-200）
