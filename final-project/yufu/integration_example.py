"""
整合範例：在 testPlayerPoseEst.py 中使用訓練好的球種分類器

這個檔案展示如何將訓練好的 AI 模型整合到主程式中
"""

# ==================== 在檔案開頭加入 ====================
from shot_classifier_inference import ShotClassifierInference
import os

# ==================== 在 main() 函數開頭初始化 ====================
def main():
    # ... 現有的初始化代碼 ...
    
    # === 初始化球種分類器（AI 模型）===
    USE_AI_CLASSIFIER = True  # 是否使用 AI 分類器
    shot_classifier = None
    
    if USE_AI_CLASSIFIER:
        weights_dir = './shot_classifier_weights'
        if os.path.exists(weights_dir):
            # 尋找最新的模型
            model_files = [f for f in os.listdir(weights_dir) 
                          if f.startswith('best_model_') and f.endswith('.pth')]
            if model_files:
                model_files.sort()
                model_path = os.path.join(weights_dir, model_files[-1])
                scaler_path = os.path.join(weights_dir, 'scaler.pkl')
                
                try:
                    shot_classifier = ShotClassifierInference(
                        model_path=model_path,
                        scaler_path=scaler_path,
                        device=device,  # 使用與 YOLO 相同的 device
                        model_type='light'
                    )
                    print("✅ AI 球種分類器已載入")
                except Exception as e:
                    print(f"⚠️  AI 分類器載入失敗: {e}")
                    print("   將使用規則判斷")
                    USE_AI_CLASSIFIER = False
            else:
                print("⚠️  找不到訓練好的模型，將使用規則判斷")
                USE_AI_CLASSIFIER = False
        else:
            print("⚠️  找不到模型目錄，將使用規則判斷")
            USE_AI_CLASSIFIER = False


# ==================== 在偵測到擊球時使用 AI 分類器 ====================
# 在原本的球種判斷代碼後面加入：

                    # === ⭐ 使用 AI 分類器（如果已載入）===
                    ai_predicted = None
                    ai_confidence = 0
                    
                    if USE_AI_CLASSIFIER and shot_classifier is not None:
                        # 準備參數字典
                        ai_params = {
                            'overall_slope': overall_slope,
                            'highest_position_ratio': highest_position_ratio,
                            'velocity': velocity,
                            'acceleration': acceleration,
                            'y_range': y_range,
                            'high_ball_ratio': high_ball_ratio,
                            'last_frames_low': last_frames_low,
                            'has_turning_point': has_turning_point
                        }
                        
                        try:
                            ai_predicted, ai_confidence, ai_probs = shot_classifier.predict(
                                ai_params,
                                return_probabilities=True
                            )
                            
                            print(f"\n🤖 AI 預測結果:")
                            print(f"   預測: {ai_predicted} (信心度: {ai_confidence:.2%})")
                            print(f"   機率分布:")
                            for label, prob in ai_probs.items():
                                bar = "█" * int(prob * 20)
                                print(f"      {label:6s}: {bar} {prob:.2%}")
                            
                            # 如果 AI 信心度高，可以優先使用 AI 的判斷
                            if ai_confidence > 0.5:  # 信心度超過 50%
                                print(f"   ✅ AI 信心度高，使用 AI 判斷")
                                shot_type = ai_predicted
                            else:
                                print(f"   ⚠️  AI 信心度低，使用規則判斷")
                        
                        except Exception as e:
                            print(f"   ❌ AI 預測失敗: {e}")
                    
                    # === 顯示最終結果 ===
                    print(f"\n" + "="*60)
                    print(f"📊 最終判斷結果")
                    print(f"="*60)
                    print(f"   規則判斷: {shot_type}")
                    if ai_predicted:
                        print(f"   AI 判斷: {ai_predicted} (信心度: {ai_confidence:.2%})")
                    print(f"="*60)


# ==================== 完整的判斷流程 ====================
"""
建議的判斷策略：

1. 並行顯示（收集資料階段）
   - 同時顯示規則判斷和 AI 判斷
   - 讓使用者標註時可以對比兩種結果
   - 持續收集更多標註資料

2. AI 優先（資料充足後）
   - 當 AI 信心度 > 70%：使用 AI 判斷
   - 當 AI 信心度 50-70%：優先 AI，但顯示警告
   - 當 AI 信心度 < 50%：使用規則判斷

3. 混合策略
   - 某些明顯案例用規則（如明顯向上飛 = Clear）
   - 模糊案例用 AI
   - 持續學習改進
"""


# ==================== 儲存標註時同時記錄 AI 預測 ====================
# 修改 save_shot_record 函數：

def save_shot_record(frame_idx, params, predicted_type, user_label=None, 
                     ai_predicted=None, ai_confidence=None):
    """儲存擊球記錄（含 AI 預測）"""
    history = load_shot_history()
    
    record = {
        'timestamp': datetime.now().isoformat(),
        'frame': frame_idx,
        'parameters': params,
        'predicted': predicted_type,  # 規則判斷
        'user_label': user_label,
        'correct': (predicted_type == user_label) if user_label else None
    }
    
    # 加入 AI 預測資訊
    if ai_predicted is not None:
        record['ai_predicted'] = ai_predicted
        record['ai_confidence'] = float(ai_confidence) if ai_confidence else 0
        record['ai_correct'] = (ai_predicted == user_label) if user_label else None
    
    history.append(record)
    
    with open(SHOT_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ==================== 分析準確率 ====================
"""
訓練更多資料後，可以分析規則 vs AI 的準確率：

import json
data = json.load(open('shot_annotations.json'))
labeled = [d for d in data if 'user_label' in d and d['user_label']]

# 規則判斷準確率
rule_correct = sum(1 for d in labeled if d.get('correct'))
rule_acc = rule_correct / len(labeled) * 100

# AI 判斷準確率
ai_correct = sum(1 for d in labeled if d.get('ai_correct'))
ai_acc = ai_correct / len(labeled) * 100

print(f"規則準確率: {rule_acc:.1f}%")
print(f"AI 準確率: {ai_acc:.1f}%")
"""

print("""
✅ 整合指南已完成！

下一步：
1. 將上述代碼整合到 testPlayerPoseEst.py
2. 執行程式，對比規則判斷和 AI 判斷
3. 持續收集更多標註資料（目標：100+ 樣本）
4. 重新訓練模型提高準確率
5. 根據準確率調整使用策略

目前狀況：
- 訓練樣本：22 個（太少！）
- 測試準確率：60%（需要更多資料）
- 建議：先並行顯示，收集 50+ 樣本後再優先使用 AI
""")
