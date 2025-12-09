# ✅ 文件更新完成

## 🔄 已更新的文件

### 1. SHUTTLECOCK_TRAINING_GUIDE.md
**修改內容：**
- ✅ 移除 labelImg 安裝說明
- ✅ 改用 `simple_annotator.py` 作為標註工具
- ✅ 更新操作說明和快捷鍵說明
- ✅ 移除 labelImg 特定的設定步驟（YOLO 格式切換、Save Dir 設定等）

**新的標註指令：**
```bash
# 標註 train
python simple_annotator.py train

# 標註 val
python simple_annotator.py val
```

---

### 2. improve_shuttlecock_detection.py
**修改內容：**
- ✅ 更新所有 labelImg 提示為 simple_annotator.py
- ✅ 修改完成訊息，加入 merge 步驟
- ✅ 更新 full mode 的工作流程說明

**新的工作流程：**
```bash
# 1. 擷取圖片
python improve_shuttlecock_detection.py --mode extract --num-frames 400

# 2. 合併到訓練集
python improve_shuttlecock_detection.py --mode merge

# 3. 標註圖片
python simple_annotator.py train
python simple_annotator.py val

# 4. 訓練模型
python improve_shuttlecock_detection.py --mode train --epochs 100
```

---

### 3. extract_frames_for_annotation.py
**修改內容：**
- ✅ 更新提示訊息使用 simple_annotator.py

---

### 4. train_shuttlecock_detector.py
**修改內容：**
- ✅ 更新檢查訊息使用 simple_annotator.py

---

### 5. ANNOTATION_WORKFLOW.md (新建)
**內容：**
- ✅ 完整的標註工作流程（從 334 張圖片開始）
- ✅ simple_annotator.py 詳細操作說明
- ✅ 標註重點和注意事項
- ✅ 訓練和使用新模型的步驟
- ✅ 故障排除指南

---

## 🎯 下一步行動

### 立即執行：
```bash
cd "/Users/eric/Documents/UW 修課/2025 fall/EEP 596A CV/EEP-596-Computer-Vision/final-project/yufu"

# 1. 合併已擷取的 334 張圖片到 train/val
python improve_shuttlecock_detection.py --mode merge

# 2. 開始標註訓練集
python simple_annotator.py train
```

---

## 📊 當前進度

| 步驟 | 狀態 | 詳細 |
|------|------|------|
| 擷取圖片 | ✅ 完成 | 334 張圖片在 `additional/` |
| 合併資料 | ⏳ 待執行 | `--mode merge` |
| 標註 train | ⏳ 待執行 | `simple_annotator.py train` |
| 標註 val | ⏳ 待執行 | `simple_annotator.py val` |
| 訓練模型 | ⏳ 待執行 | `--mode train --epochs 100` |
| 更新模型路徑 | ⏳ 待執行 | 修改 `testPlayerPoseEst.py` |

---

## 🔧 simple_annotator.py 優勢

相比 labelImg：
- ✅ 無需安裝（已包含在專案中）
- ✅ Python 3.13 兼容（labelImg 有問題）
- ✅ 更簡單的操作（只有必要功能）
- ✅ 自動 YOLO 格式（無需設定）
- ✅ 自動保存（按 SPACE 即可）
- ✅ 支援多框標註（一張圖多個球）
- ✅ 可載入已有標註（繼續標註）

---

## 📖 操作快速參考

### simple_annotator.py 快捷鍵
```
滑鼠左鍵拖曳  → 畫標註框
SPACE        → 儲存並下一張 ✅
D            → 跳過此圖 ⏭️
A            → 回上一張 ⏮️
R            → 重新標註 🔄
Q            → 結束 👋
```

### 標註品質檢查清單
- [ ] 框緊貼球體（不要太大）
- [ ] 只標註清楚的球（模糊的跳過）
- [ ] 所有清楚的球都有標（不要漏標）
- [ ] train 和 val 標準一致

---

## ⚠️ 重要提醒

1. **先執行 merge**
   - 必須先將 additional 圖片合併到 train/val
   - simple_annotator.py 會直接讀取 train/val 目錄

2. **標註可以中斷**
   - 按 Q 隨時退出
   - 已標註的會自動保存
   - 下次執行會載入已有標註

3. **檢查標註數量**
   ```bash
   # 查看已標註數量
   ls badminton_ball_dataset/labels/train/*.txt | wc -l
   ls badminton_ball_dataset/labels/val/*.txt | wc -l
   ```

---

## 🎉 預期結果

標註並訓練完成後：
- 偵測率提升（找到更多之前漏掉的球）
- 誤判減少（訓練資料更多樣）
- 不同光線條件下更穩定
- 整體追蹤更流暢
