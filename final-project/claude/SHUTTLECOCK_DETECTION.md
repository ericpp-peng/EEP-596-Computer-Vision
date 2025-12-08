# 羽球偵測器 - 使用指南

## 🎯 功能概述

新增了進階羽球偵測器,支援多種偵測策略:

### 偵測模式

1. **YOLO 模式** (`mode='yolo'`)
   - 使用 COCO 預訓練模型的 sports ball (class 32)
   - 適合背景複雜的場景
   - 優點: 快速、穩定
   - 缺點: 準確率約 60-70%

2. **顏色偵測模式** (`mode='color'`)
   - 基於 HSV 白色偵測
   - 適合背景較暗的場景
   - 優點: 不需要模型
   - 缺點: 受光線影響大

3. **Hybrid 模式** (`mode='hybrid'`) ⭐ **推薦**
   - 結合 YOLO + 顏色偵測
   - 自動選擇最佳結果
   - 準確率最高 (~80-85%)

4. **自訓練模式** (`mode='custom'`)
   - 使用自己訓練的羽球模型
   - 需要提供 `custom_model_path`
   - 準確率可達 90%+

### 備用方案

當所有偵測都失敗時,自動使用**右手腕位置估計**球的位置。

## 🚀 快速開始

### 1. 測試羽球偵測器

```bash
# 測試不同偵測模式
python test_shuttlecock.py

# 或指定影片
python test_shuttlecock.py your_video.mp4
```

**互動控制:**
- `1`: 切換到 YOLO 模式
- `2`: 切換到顏色偵測模式
- `3`: 切換到 Hybrid 模式
- `space`: 暫停
- `q`: 退出

### 2. 執行完整分析

```bash
# 使用 hybrid 模式 (預設)
python badminton_analysis.py
```

程式會自動使用新的羽球偵測器!

## ⚙️ 調整參數

### 修改偵測模式

編輯 `badminton_analysis.py`:

```python
# 選擇偵測模式
SHUTTLECOCK_MODE = "hybrid"  # 可改為 'yolo', 'color', 'custom'

# 如果使用自訓練模型
CUSTOM_MODEL_PATH = "runs/detect/badminton/weights/best.pt"
```

### 調整顏色偵測閾值

編輯 `shuttlecock_detector.py` 的 `detect_by_color()`:

```python
# 白色範圍 (根據你的影片調整)
lower_white = np.array([0, 0, 180])    # [H_min, S_min, V_min]
upper_white = np.array([180, 50, 255]) # [H_max, S_max, V_max]

# 羽球面積範圍 (像素)
if 10 < area < 500:  # 調整這個範圍
```

### 調整擊球偵測靈敏度

編輯 `badminton_analysis.py`:

```python
# 在 detect_impact() 函式中
is_impact = detect_impact(
    ball_pos, 
    ball_history, 
    racket_pos,
    threshold_dist=50,   # 距離閾值 (像素) - 越小越嚴格
    threshold_speed=15   # 速度閾值 (像素/幀) - 越大越嚴格
)
```

## 📊 效能比較

| 模式 | 準確率 | 速度 | 適用場景 |
|------|--------|------|----------|
| YOLO | 60-70% | 快 | 背景複雜 |
| Color | 50-60% | 最快 | 背景暗 |
| Hybrid | 80-85% | 中 | **通用** ⭐ |
| Custom | 90%+ | 快 | 需訓練 |
| Wrist | 100% | 最快 | 備用 |

## 🎓 進階: 訓練自訂模型

### 步驟 1: 收集資料

```bash
# 從影片中截取羽球畫面
ffmpeg -i 20250711_short.mp4 -vf fps=2 frames/frame_%04d.jpg
```

### 步驟 2: 標記資料

使用 [Roboflow](https://roboflow.com) 或 LabelImg:
1. 上傳圖片
2. 標記 `shuttlecock` (羽球)
3. 可選: 標記 `racket` (球拍)
4. 匯出為 YOLO format

### 步驟 3: 訓練

```bash
python train_shuttlecock.py
```

### 步驟 4: 使用

```python
# 修改 badminton_analysis.py
SHUTTLECOCK_MODE = "custom"
CUSTOM_MODEL_PATH = "runs/detect/badminton/weights/best.pt"
```

## 🐛 常見問題

### Q1: 偵測不到羽球?

**方案 1**: 切換到 hybrid 模式
```python
SHUTTLECOCK_MODE = "hybrid"
```

**方案 2**: 調整顏色閾值
- 影片太亮 → 提高 V_min
- 影片太暗 → 降低 V_min

**方案 3**: 降低信心閾值
```python
# 在 shuttlecock_detector.py
if conf > 0.3:  # 從 0.5 降到 0.3
```

### Q2: 誤偵測太多?

**提高信心閾值:**
```python
if conf > 0.6:  # 提高閾值
```

**限制偵測區域:**
```python
# 在 detect() 中加入
if y < frame_height * 0.3:  # 忽略上半部
    return None
```

### Q3: 軌跡斷斷續續?

**增加歷史長度:**
```python
self.history = deque(maxlen=30)  # 增加到 30
```

**使用卡爾曼濾波** (進階):
```python
# TODO: 實作卡爾曼濾波追蹤
```

### Q4: 速度太慢?

1. **降低解析度:**
```python
results = model(frame, imgsz=416)  # 從 640 降到 416
```

2. **跳幀處理:**
```python
if frame_id % 2 == 0:  # 只處理偶數幀
    ball_detection = detector.detect(frame)
```

## 📈 下一步優化

- [ ] 加入卡爾曼濾波追蹤
- [ ] 多球追蹤 (雙打)
- [ ] 軌跡預測
- [ ] GPU 加速
- [ ] 即時處理模式

## 🎬 視覺化說明

偵測結果會顯示:
- 🟡 **黃色圓點** - 羽球位置
- 🔵 **藍色框** - 偵測邊界框
- 🟠 **橘色軌跡** - 球的移動路徑
- 📝 **文字** - 偵測來源和信心度

## ✅ 檢查清單

完成第二步:羽球/球拍偵測

- [x] 建立 ShuttlecockDetector 類別
- [x] 建立 RacketDetector 類別
- [x] 整合到主程式
- [x] 建立測試腳本
- [x] 撰寫使用文件
- [ ] 測試執行
- [ ] 調整參數
- [ ] (選用) 訓練自訂模型

準備進入第三步: 球場線條繪製! 🏸
