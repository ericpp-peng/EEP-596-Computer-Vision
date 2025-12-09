# 🚀 球種分類 AI 系統 - 快速入門

## ✅ 系統已建立完成！

你現在擁有一個完整的球種分類訓練系統，可以從標註資料訓練 AI 模型來自動判斷羽球擊球類型。

## 📁 重要檔案一覽

```
yufu/
├── 📘 SHOT_CLASSIFIER_README.md          # 詳細使用說明
├── 📘 TRAINING_COMPLETE_REPORT.md        # 訓練完成報告
├── 📘 QUICK_START.md                     # 本檔案
│
├── 🔧 shot_classifier_model.py           # 模型定義
├── 🔧 shot_dataset.py                    # 資料集處理
├── 🔧 train_shot_classifier.py           # 訓練腳本
├── 🔧 shot_classifier_inference.py       # 推理模組
├── 🔧 integration_example.py             # 整合範例
├── 🔧 verify_system.py                   # 系統驗證工具
│
├── 📊 shot_annotations.json              # 標註資料 (22 樣本)
│
└── 📂 shot_classifier_weights/           # 訓練產出
    ├── best_model_20251209_152327.pth   # ⭐ 最佳模型
    ├── scaler.pkl                        # ⭐ 特徵標準化
    ├── confusion_matrix.png              # 混淆矩陣
    ├── training_history.png              # 訓練曲線
    └── training_info.json                # 訓練資訊
```

## 🎯 3 步驟快速使用

### 步驟 1：驗證系統

```bash
python verify_system.py
```

應該看到所有測試通過 ✅

### 步驟 2：測試推理

```bash
python shot_classifier_inference.py
```

會顯示 AI 對幾個測試案例的預測結果。

### 步驟 3：整合到主程式

參考 `integration_example.py` 中的代碼，在 `testPlayerPoseEst.py` 中加入：

```python
from shot_classifier_inference import ShotClassifierInference

# 在 main() 開頭初始化
shot_classifier = ShotClassifierInference(
    model_path='./shot_classifier_weights/best_model_20251209_152327.pth',
    scaler_path='./shot_classifier_weights/scaler.pkl',
    device='mps',
    model_type='light'
)

# 在偵測到擊球時使用
ai_predicted, ai_confidence = shot_classifier.predict({
    'overall_slope': overall_slope,
    'highest_position_ratio': highest_position_ratio,
    'velocity': velocity,
    'acceleration': acceleration,
    'y_range': y_range,
    'high_ball_ratio': high_ball_ratio,
    'last_frames_low': last_frames_low,
    'has_turning_point': has_turning_point
})

print(f"AI 預測: {ai_predicted} ({ai_confidence:.2%})")
```

## 📊 目前狀態

```
✅ 系統建立完成
✅ 初步訓練完成
   - 訓練樣本：17 個
   - 測試樣本：5 個
   - 測試準確率：60%
   
⚠️  資料量不足
   - 目前：22 個樣本
   - 建議：50-100 個樣本
   
⚠️  準確率待提升
   - 目前：60%
   - 目標：80%+
```

## 🎯 下一步行動

### 優先：收集更多資料

1. 執行 `testPlayerPoseEst.py`
2. 每次偵測到擊球時標註：
   - `1` = Smash (殺球)
   - `2` = Clear (高遠球)
   - `3` = Drop (切球)
3. 目標：至少 50 個樣本

### 然後：重新訓練

```bash
python train_shot_classifier.py
```

資料越多，準確率越高！

## 💡 使用建議

### 階段 1：資料收集（目前）
- ✅ 系統已建立
- 🎯 **收集 50+ 樣本**
- 📊 並行顯示規則 + AI 判斷
- 📝 持續標註建立資料集

### 階段 2：模型改進（50+ 樣本後）
- 🔄 重新訓練
- 📈 分析錯誤案例
- 🎯 目標準確率 80%+

### 階段 3：正式部署（80%+ 準確率）
- 🚀 AI 優先使用
- 🔧 規則作為後備
- 📊 持續改進

## 🔍 常見問題

**Q: AI 準確率只有 60%，還能用嗎？**

A: 目前樣本太少（22個），建議：
- 先並行顯示規則 + AI 判斷
- 收集更多資料（50+）
- 重新訓練後準確率會大幅提升

**Q: 如何提高準確率？**

A: 
1. **最重要：收集更多資料**
2. 確保各類別樣本平衡
3. 使用不同影片（增加多樣性）
4. 檢查標註品質

**Q: 要訓練多久？**

A: 
- 目前設定：50 輪（1-2 分鐘）
- 有 Early Stopping，會自動停止
- 資料增加後可以調整輪數

**Q: 可以用 CPU 訓練嗎？**

A: 
- 可以！資料量小時 CPU 也很快
- M4 Pro 的 MPS 加速更好

## 📚 詳細文檔

- `SHOT_CLASSIFIER_README.md` - 完整使用說明
- `TRAINING_COMPLETE_REPORT.md` - 訓練報告
- `integration_example.py` - 整合代碼範例

## 🎓 技術要點

這個系統展示了：

✅ **完整 ML 流程**
- 資料收集與標註
- 特徵工程
- 模型訓練
- 評估與優化
- 部署推理

✅ **小數據處理**
- 輕量模型設計
- Dropout 防過擬合
- 類別權重平衡
- Early stopping

✅ **工程實踐**
- 模組化設計
- 可重複流程
- 完整評估
- 視覺化結果

## 🎉 恭喜！

你已經成功建立了一個完整的球種分類 AI 系統！

**下一步最重要：收集更多標註資料！**

有了足夠的資料，AI 準確率可以輕鬆超過 90%，遠勝規則判斷。

加油！🚀

---

## 快速指令參考

```bash
# 驗證系統
python verify_system.py

# 測試推理
python shot_classifier_inference.py

# 重新訓練
python train_shot_classifier.py

# 查看資料統計
python -c "import json; from collections import Counter; data = json.load(open('shot_annotations.json')); labeled = [d for d in data if 'user_label' in d]; print(f'總樣本: {len(labeled)}'); print('分布:', Counter([d['user_label'] for d in labeled]))"
```

需要幫助？查看詳細文檔或檢查各檔案的註解！
