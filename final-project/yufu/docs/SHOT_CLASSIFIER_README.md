# 球種分類訓練系統使用指南

這是一個基於神經網路的羽球擊球類型分類系統，可以根據球的軌跡參數自動判斷球種（殺球、高遠球、切球）。

## 📁 檔案結構

```
shot_classifier_model.py      # 神經網路模型定義
shot_dataset.py               # 資料集處理
train_shot_classifier.py      # 訓練腳本
shot_classifier_inference.py  # 推理模組
shot_annotations.json         # 標註資料（由 testPlayerPoseEst.py 產生）
shot_classifier_weights/      # 訓練權重儲存目錄
    ├── best_model_*.pth      # 最佳模型權重
    ├── final_model_*.pth     # 最終模型權重
    ├── scaler.pkl            # 特徵標準化參數
    ├── confusion_matrix.png  # 混淆矩陣
    ├── training_history.png  # 訓練歷史圖表
    └── training_info.json    # 訓練資訊
```

## 🚀 使用流程

### 1. 收集標註資料

執行 `testPlayerPoseEst.py`，在偵測到擊球時按鍵標註：
- `1` - Smash (殺球)
- `2` - Clear (高遠球)  
- `3` - Drop (切球)
- `s` - 跳過

標註資料會自動儲存到 `shot_annotations.json`。

**建議至少收集 50+ 個樣本** (目前只有 22 個)

### 2. 訓練模型

```bash
python train_shot_classifier.py
```

訓練參數（可在 `train_shot_classifier.py` 中調整）：
- `NUM_EPOCHS`: 訓練輪數（預設 150）
- `LEARNING_RATE`: 學習率（預設 0.001）
- `BATCH_SIZE`: batch 大小（預設 4，小數據集用小 batch）
- `USE_LIGHT_MODEL`: 是否使用輕量模型（預設 True）

### 3. 測試推理

```bash
python shot_classifier_inference.py
```

會測試幾個典型案例並顯示預測結果。

### 4. 整合到主程式

在 `testPlayerPoseEst.py` 中整合訓練好的模型：

```python
from shot_classifier_inference import ShotClassifierInference

# 初始化分類器（在 main() 開頭）
shot_classifier = ShotClassifierInference(
    model_path='./shot_classifier_weights/best_model_XXX.pth',
    scaler_path='./shot_classifier_weights/scaler.pkl',
    device='mps',  # 或 'cuda', 'cpu'
    model_type='light'
)

# 使用分類器（在偵測到擊球時）
params = {
    'overall_slope': overall_slope,
    'highest_position_ratio': highest_position_ratio,
    'velocity': velocity,
    'acceleration': acceleration,
    'y_range': y_range,
    'high_ball_ratio': high_ball_ratio,
    'last_frames_low': last_frames_low,
    'has_turning_point': has_turning_point
}

predicted_class, confidence = shot_classifier.predict(params)
print(f"AI 預測: {predicted_class} (信心度: {confidence:.2%})")
```

## 📊 模型架構

### 輸入特徵（8 個）
1. `overall_slope` - 整體斜率（球的上下移動趨勢）
2. `highest_position_ratio` - 最高點位置比例（0-1）
3. `velocity` - 平均速度（pixels/s）
4. `acceleration` - 加速度
5. `y_range` - 垂直移動範圍
6. `high_ball_ratio` - 高處停留比例
7. `last_frames_low` - 最後幾幀是否在低處（0/1）
8. `has_turning_point` - 是否有轉折點（0/1）

### 輸出
3 個類別的機率分布：Smash, Clear, Drop

### 網路結構（輕量模型）
```
Input (8) → FC(32) → BatchNorm → ReLU → Dropout 
         → FC(16) → BatchNorm → ReLU → Dropout
         → FC(3) → Softmax
```

總參數量：約 700 個參數

## 📈 訓練技巧

### 處理小數據集
- ✅ 使用輕量模型（避免過擬合）
- ✅ 加入 Dropout（0.2）
- ✅ 使用 BatchNorm
- ✅ 類別權重平衡
- ✅ Early Stopping
- ✅ 學習率調度

### 收集更多資料
目前只有 22 個樣本，建議：
1. 多標註不同的影片
2. 標註不同球員的擊球
3. 確保各類別平衡（目前 Clear:10, Drop:7, Smash:5）

## 🔧 進階調整

### 修改模型結構
編輯 `shot_classifier_model.py` 中的 `LightShotClassifier` 或 `ShotClassifier`

### 修改訓練參數
編輯 `train_shot_classifier.py` 中的參數設定區

### 新增特徵
在 `shot_dataset.py` 的 `extract_features()` 函數中新增

## 📊 評估指標

訓練完成後會產生：
1. **混淆矩陣** (`confusion_matrix.png`) - 查看分類錯誤模式
2. **訓練曲線** (`training_history.png`) - 查看訓練/驗證 loss 和 accuracy
3. **分類報告** - Precision, Recall, F1-score
4. **訓練資訊** (`training_info.json`) - 完整訓練記錄

## ⚠️ 目前限制

1. **樣本數太少**（22 個）- 建議至少 50-100 個
2. **類別不平衡** - Smash 只有 5 個樣本
3. **模型可能過擬合** - 需要更多資料驗證

## 💡 建議下一步

1. ✨ **收集更多標註資料**（最重要！）
2. 🎯 確保各類別樣本平衡
3. 📹 使用多個不同影片（增加多樣性）
4. 🔬 分析錯誤案例，調整特徵或門檻
5. 🚀 整合到主程式並持續改進

## 🆚 AI vs 規則判斷

### 規則判斷（目前）
- ✅ 簡單直觀
- ✅ 可解釋性強
- ❌ 需要手動調整門檻
- ❌ 難以處理邊界案例

### AI 判斷（訓練後）
- ✅ 自動學習特徵組合
- ✅ 適應性強
- ✅ 可持續改進
- ❌ 需要標註資料
- ❌ 黑盒子（較難解釋）

**建議**：初期可以並行使用，顯示兩種判斷結果對比，持續收集資料改進 AI 模型。

---

有問題請參考各檔案的註解或執行測試腳本！
