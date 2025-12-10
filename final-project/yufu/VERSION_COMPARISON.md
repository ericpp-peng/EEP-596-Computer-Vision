# 程式版本說明

## 📚 兩個版本的差異

### 1️⃣ `testPlayerPoseEst.py` - **標註訓練版本**

**用途：** 用來標註答案、訓練模型、收集數據

**特點：**
- ✅ 偵測到擊球後會**暫停**
- ✅ 等待用戶按鍵標註正確答案：
  - `C` = Clear (高遠球)
  - `S` = Smash (殺球)
  - `D` = Drop (切球)
  - `Space` = 接受 AI 判斷
- ✅ 可以按 `P` 暫停影片
- ✅ 記錄用戶標註與 AI 預測的對比
- ✅ 用於改進模型準確度

**使用時機：**
- 需要訓練或改進模型
- 想要驗證 AI 預測的準確性
- 收集標註數據

---

### 2️⃣ `testPlayerPoseEst_auto.py` - **自動預測版本**

**用途：** 完全自動處理，展示系統預測能力

**特點：**
- 🤖 完全自動，**不需要任何用戶輸入**
- 🚀 從頭到尾連續處理影片
- 📊 自動接受 AI 的預測結果
- 🎬 輸出檔案：`output_with_pose_auto.mp4`
- ⚡ 沒有暫停功能（按 `Q` 可中斷）

**使用時機：**
- 快速預覽系統效果
- 處理大量影片
- 展示給他人看
- 不想手動標註

---

## 🚀 使用方法

### 標註訓練版本
```bash
# 使用預設影片
python testPlayerPoseEst.py

# 指定輸入影片
python testPlayerPoseEst.py -i my_video.mp4

# 指定輸入和輸出影片
python testPlayerPoseEst.py -i input.mp4 -o output.mp4

# 查看幫助
python testPlayerPoseEst.py --help
```
偵測到擊球時會暫停，按對應按鍵標註答案。

### 自動預測版本
```bash
# 使用預設影片
python testPlayerPoseEst_auto.py

# 指定輸入影片
python testPlayerPoseEst_auto.py -i my_video.mp4

# 指定輸入和輸出影片
python testPlayerPoseEst_auto.py -i input.mp4 -o output.mp4

# 查看幫助
python testPlayerPoseEst_auto.py --help
```
完全自動運行，不需要任何操作。

---

## 🔧 核心邏輯

兩個版本使用**完全相同的**：
- ✅ 球偵測算法（YOLO + 差分）
- ✅ 軌跡追蹤邏輯
- ✅ 擊球分類決策樹
- ✅ 所有參數和門檻值
- ✅ AI 建議系統

**唯一差異**：用戶互動方式
- 訓練版：需要確認答案
- 自動版：直接接受預測

---

## 📝 建議使用流程

1. **初期開發階段** → 使用 `testPlayerPoseEst.py`
   - 標註數據
   - 訓練模型
   - 調整參數

2. **系統穩定後** → 使用 `testPlayerPoseEst_auto.py`
   - 快速處理影片
   - 展示系統能力
   - 批量分析

3. **持續改進** → 定期回到 `testPlayerPoseEst.py`
   - 收集錯誤案例
   - 重新訓練
   - 提升準確度
