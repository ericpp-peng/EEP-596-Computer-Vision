#!/usr/bin/env python
"""
快速驗證球種分類系統

這個腳本會：
1. 測試資料集載入
2. 測試模型定義
3. 載入訓練好的模型
4. 進行預測測試
5. 顯示系統狀態
"""

import os
import sys
import json

def print_section(title):
    """打印分隔線"""
    print("\n" + "=" * 60)
    print(f"📌 {title}")
    print("=" * 60)

def check_files():
    """檢查必要檔案是否存在"""
    print_section("檢查檔案")
    
    required_files = [
        'shot_classifier_model.py',
        'shot_dataset.py',
        'train_shot_classifier.py',
        'shot_classifier_inference.py',
        'shot_annotations.json'
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        all_exist = all_exist and exists
    
    return all_exist

def check_weights():
    """檢查訓練權重"""
    print_section("檢查訓練權重")
    
    weights_dir = './shot_classifier_weights'
    if not os.path.exists(weights_dir):
        print("❌ 權重目錄不存在")
        return False
    
    files = os.listdir(weights_dir)
    model_files = [f for f in files if f.endswith('.pth')]
    
    print(f"📁 權重目錄: {weights_dir}")
    print(f"📊 找到 {len(model_files)} 個模型檔案")
    
    for f in files:
        size = os.path.getsize(os.path.join(weights_dir, f))
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024*1024:
            size_str = f"{size/1024:.1f}KB"
        else:
            size_str = f"{size/1024/1024:.1f}MB"
        print(f"   {f}: {size_str}")
    
    return len(model_files) > 0

def check_data():
    """檢查標註資料"""
    print_section("檢查標註資料")
    
    try:
        with open('shot_annotations.json', 'r') as f:
            data = json.load(f)
        
        labeled = [d for d in data if 'user_label' in d and d['user_label']]
        
        print(f"📊 總樣本數: {len(data)}")
        print(f"📝 已標註: {len(labeled)}")
        
        # 統計各類別
        from collections import Counter
        labels = [d['user_label'] for d in labeled]
        counts = Counter(labels)
        
        print(f"\n類別分布:")
        for label, count in counts.most_common():
            bar = "█" * (count * 2)
            print(f"   {label:6s}: {bar} {count}")
        
        # 檢查資料品質
        print(f"\n資料品質:")
        if len(labeled) < 30:
            print(f"   ⚠️  樣本數較少 ({len(labeled)})，建議至少 50 個")
        else:
            print(f"   ✅ 樣本數充足")
        
        # 檢查平衡性
        min_count = min(counts.values())
        max_count = max(counts.values())
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 3:
            print(f"   ⚠️  類別不平衡（比例 {imbalance_ratio:.1f}:1）")
        else:
            print(f"   ✅ 類別相對平衡")
        
        return len(labeled) > 0
        
    except Exception as e:
        print(f"❌ 讀取資料失敗: {e}")
        return False

def test_model():
    """測試模型"""
    print_section("測試模型")
    
    try:
        import torch
        from shot_classifier_model import LightShotClassifier
        
        model = LightShotClassifier(input_dim=8, num_classes=3)
        
        # 計算參數數量
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型建立成功")
        print(f"📊 參數數量: {params:,}")
        
        # 測試前向傳播
        x = torch.randn(4, 8)
        output = model(x)
        print(f"📈 輸出形狀: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 模型測試失敗: {e}")
        return False

def test_inference():
    """測試推理"""
    print_section("測試推理")
    
    try:
        from shot_classifier_inference import ShotClassifierInference
        import torch
        
        # 尋找最新模型
        weights_dir = './shot_classifier_weights'
        model_files = [f for f in os.listdir(weights_dir) 
                      if f.startswith('best_model_') and f.endswith('.pth')]
        
        if not model_files:
            print("❌ 找不到訓練好的模型")
            return False
        
        model_files.sort()
        model_path = os.path.join(weights_dir, model_files[-1])
        scaler_path = os.path.join(weights_dir, 'scaler.pkl')
        
        # 建立推理器
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        classifier = ShotClassifierInference(
            model_path=model_path,
            scaler_path=scaler_path,
            device=device,
            model_type='light'
        )
        
        print(f"✅ 推理器建立成功")
        
        # 測試預測
        test_params = {
            'overall_slope': 200.0,
            'highest_position_ratio': 0.1,
            'velocity': 300.0,
            'acceleration': 5.0,
            'y_range': 200.0,
            'high_ball_ratio': 0.3,
            'last_frames_low': True,
            'has_turning_point': False
        }
        
        predicted, confidence = classifier.predict(test_params)
        print(f"📊 測試預測: {predicted} (信心度: {confidence:.2%})")
        
        return True
        
    except Exception as e:
        print(f"❌ 推理測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_training_info():
    """顯示訓練資訊"""
    print_section("訓練資訊")
    
    info_path = './shot_classifier_weights/training_info.json'
    if not os.path.exists(info_path):
        print("❌ 找不到訓練資訊")
        return
    
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(f"🕐 訓練時間: {info.get('timestamp', 'N/A')}")
        print(f"🔢 訓練輪數: {info.get('num_epochs', 'N/A')}")
        print(f"📊 訓練樣本: {info.get('train_samples', 'N/A')}")
        print(f"📊 測試樣本: {info.get('test_samples', 'N/A')}")
        print(f"🎯 最佳驗證準確率: {info.get('best_val_acc', 'N/A'):.2f}%")
        print(f"🎯 測試準確率: {info.get('test_acc', 'N/A'):.2f}%")
        print(f"📈 最終訓練準確率: {info.get('final_train_acc', 'N/A'):.2f}%")
        
        print(f"\n類別分布:")
        for label, count in info.get('label_counts', {}).items():
            print(f"   {label}: {count}")
        
    except Exception as e:
        print(f"❌ 讀取訓練資訊失敗: {e}")

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║         球種分類系統驗證工具                              ║
║         Shot Classifier System Verification              ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # 1. 檢查檔案
    results.append(("檔案檢查", check_files()))
    
    # 2. 檢查權重
    results.append(("權重檢查", check_weights()))
    
    # 3. 檢查資料
    results.append(("資料檢查", check_data()))
    
    # 4. 測試模型
    results.append(("模型測試", test_model()))
    
    # 5. 測試推理
    results.append(("推理測試", test_inference()))
    
    # 6. 顯示訓練資訊
    show_training_info()
    
    # 總結
    print_section("總結")
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "🎉" * 30)
        print("✅ 所有測試通過！系統已準備就緒！")
        print("🎉" * 30)
        print("\n下一步：")
        print("1. 收集更多標註資料（目標 50+ 樣本）")
        print("2. 整合到 testPlayerPoseEst.py")
        print("3. 重新訓練以提高準確率")
    else:
        print("\n" + "⚠️ " * 30)
        print("❌ 部分測試失敗，請檢查錯誤訊息")
        print("⚠️ " * 30)
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n中斷執行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
