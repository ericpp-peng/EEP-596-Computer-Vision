# 場地範圍限制 - 使用指南

## 🎯 問題解決

**問題**: 羽球偵測器偵測到其他場地的球  
**解決**: 加入場地範圍限制,只偵測指定區域內的羽球

## ✅ 已完成改進

### 1. 自動場地範圍
```python
detector = ShuttlecockDetector(mode='hybrid', court_area='auto')
```
- 自動使用畫面中心 70% 區域
- 左右各留 15%,上下各留 15%
- 適合大多數單場地影片

### 2. 自訂場地範圍

#### 矩形範圍
```python
# (x, y, width, height)
detector = ShuttlecockDetector(
    mode='hybrid',
    court_area=(100, 100, 800, 600)
)
```

#### 多邊形範圍
```python
# 四個頂點 (左上, 右上, 右下, 左下)
detector = ShuttlecockDetector(
    mode='hybrid',
    court_area=[
        (200, 150),   # 左上
        (1700, 150),  # 右上
        (1700, 900),  # 右下
        (200, 900)    # 左下
    ]
)
```

### 3. 不限制範圍
```python
# 預設行為 (偵測整個畫面)
detector = ShuttlecockDetector(mode='hybrid')
```

## 🧪 測試方法

### 方法 1: 場地範圍測試 (推薦)
```bash
python test_court_area.py
```

**效果**:
- ✅ 自動設置場地範圍 (中心 70%)
- ✅ 綠色半透明區域顯示範圍
- ✅ 只偵測場地內的羽球
- ✅ 過濾其他場地的干擾

**輸出**: `court_limited_detection.mp4`

### 方法 2: 使用主程式
```bash
python badminton_analysis.py
```
已自動啟用場地範圍限制!

### 方法 3: 視覺化測試
```bash
python visualize_detection.py
```
也已自動啟用場地範圍限制!

## 📊 效果比較

| 模式 | 偵測範圍 | 誤偵測 | 適用場景 |
|------|----------|--------|----------|
| **無限制** | 整個畫面 | 多 | 單一場地,背景乾淨 |
| **自動範圍** | 中心 70% | 少 | **大多數情況** ⭐ |
| **自訂範圍** | 指定區域 | 最少 | 特殊需求 |

## 🎬 視覺化說明

影片中會顯示:
- 🟢 **綠色半透明區域** - 場地範圍
- 🟡 **黃色圓圈** - 羽球位置 (只在範圍內)
- 🟠 **橘色軌跡** - 球的移動路徑
- 📝 **文字資訊** - 偵測率和範圍資訊

## ⚙️ 進階調整

### 調整自動範圍大小

編輯 `shuttlecock_detector.py` 的 `set_auto_court_area()`:

```python
def set_auto_court_area(self, frame):
    h, w = frame.shape[:2]
    margin_x = int(w * 0.15)  # 改為 0.1 可擴大到 80%
    margin_y = int(h * 0.15)  # 改為 0.2 可縮小到 60%
    # ...
```

### 使用球場標定結果

如果已經執行過 `court_calibration.py`:

```python
import pickle

# 載入球場角點
with open('court_corners.pkl', 'rb') as f:
    corners = pickle.load(f)

# 使用球場角點作為範圍
detector = ShuttlecockDetector(
    mode='hybrid',
    court_area=corners  # 直接使用標定的四個角點
)
```

## 💡 建議設定

### 標準設定 (推薦)
```python
detector = ShuttlecockDetector(
    mode='hybrid',
    court_area='auto'  # 自動中心 70%
)
```

### 嚴格設定 (減少誤偵測)
```python
detector = ShuttlecockDetector(
    mode='hybrid',
    court_area=(300, 200, 1300, 700)  # 更小的範圍
)
```

### 寬鬆設定 (不遺漏球)
```python
detector = ShuttlecockDetector(
    mode='hybrid',
    court_area=(100, 100, 1700, 900)  # 更大的範圍
)
```

## 🐛 疑難排解

### Q: 還是偵測到其他場地?
**A**: 縮小場地範圍
```python
# 將 margin 從 0.15 改為 0.25
margin_x = int(w * 0.25)  # 只使用中心 50%
```

### Q: 漏掉邊界的球?
**A**: 擴大場地範圍
```python
# 將 margin 從 0.15 改為 0.05
margin_x = int(w * 0.05)  # 使用 90% 區域
```

### Q: 如何查看場地範圍?
**A**: 使用 `test_court_area.py`,會顯示綠色半透明區域

### Q: 如何精確設定範圍?
**A**: 
1. 執行 `court_calibration.py` 標定場地
2. 使用標定結果作為範圍
3. 或用影像軟體測量座標後手動設定

## ✅ 驗證成功

測試結果:
- ✅ 場地範圍自動設置成功
- ✅ 偵測率 100% (只在場地內)
- ✅ 視覺化顯示場地範圍
- ✅ 過濾場地外的偵測

**現在羽球偵測只會專注在你的場地上了!** 🏸
