"""
球種分類推理模組
載入訓練好的模型進行即時球種分類
"""

import torch
import numpy as np
import pickle
from shot_classifier_model import ShotClassifier, LightShotClassifier
from shot_dataset import LABEL_NAMES


class ShotClassifierInference:
    """球種分類推理器"""
    
    def __init__(self, model_path, scaler_path, device='cpu', model_type='light'):
        """
        Args:
            model_path: 模型權重路徑
            scaler_path: StandardScaler 路徑
            device: 'cpu', 'cuda', or 'mps'
            model_type: 'light' or 'standard'
        """
        self.device = device
        
        # 載入 scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # 建立模型
        if model_type == 'light':
            self.model = LightShotClassifier(input_dim=8, num_classes=3)
        else:
            self.model = ShotClassifier(input_dim=8, hidden_dims=[64, 32, 16], num_classes=3)
        
        # 載入權重
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✅ 球種分類器已載入")
        print(f"   模型: {model_path}")
        print(f"   裝置: {device}")
        if 'val_acc' in checkpoint:
            print(f"   驗證準確率: {checkpoint['val_acc']:.2f}%")
    
    def extract_features(self, params):
        """
        從參數字典提取特徵
        
        Args:
            params: dict with keys:
                - overall_slope
                - highest_position_ratio
                - velocity
                - acceleration
                - y_range
                - high_ball_ratio
                - last_frames_low (bool)
                - has_turning_point (bool)
        
        Returns:
            features: numpy array of shape (8,)
        """
        features = [
            params['overall_slope'],
            params['highest_position_ratio'],
            params['velocity'],
            params['acceleration'],
            params['y_range'],
            params['high_ball_ratio'],
            1.0 if params['last_frames_low'] else 0.0,
            1.0 if params['has_turning_point'] else 0.0
        ]
        return np.array(features, dtype=np.float32)
    
    def predict(self, params, return_probabilities=False):
        """
        預測球種
        
        Args:
            params: dict or list of features
            return_probabilities: 是否返回機率分布
        
        Returns:
            predicted_class: str (Smash/Clear/Drop)
            confidence: float (0-1)
            probabilities: dict (optional, if return_probabilities=True)
        """
        # 提取特徵
        if isinstance(params, dict):
            features = self.extract_features(params)
        else:
            features = np.array(params, dtype=np.float32)
        
        # 標準化
        features = self.scaler.transform(features.reshape(1, -1))
        
        # 轉換為 tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # 預測
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        predicted_class = LABEL_NAMES[predicted_idx]
        
        if return_probabilities:
            prob_dict = {
                LABEL_NAMES[i]: probabilities[i].item()
                for i in range(len(LABEL_NAMES))
            }
            return predicted_class, confidence, prob_dict
        else:
            return predicted_class, confidence
    
    def predict_batch(self, params_list):
        """
        批量預測
        
        Args:
            params_list: list of dicts or list of feature arrays
        
        Returns:
            predictions: list of (predicted_class, confidence)
        """
        results = []
        for params in params_list:
            predicted_class, confidence = self.predict(params)
            results.append((predicted_class, confidence))
        return results


def test_inference():
    """測試推理功能"""
    import os
    
    print("=" * 60)
    print("測試球種分類推理")
    print("=" * 60)
    
    # 尋找最新的模型
    weights_dir = './shot_classifier_weights'
    if not os.path.exists(weights_dir):
        print("❌ 找不到模型權重目錄，請先訓練模型")
        return
    
    # 找最新的 best_model
    model_files = [f for f in os.listdir(weights_dir) if f.startswith('best_model_') and f.endswith('.pth')]
    if not model_files:
        model_files = [f for f in os.listdir(weights_dir) if f.startswith('final_model_') and f.endswith('.pth')]
    
    if not model_files:
        print("❌ 找不到模型權重檔案")
        return
    
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
    
    # 測試案例
    test_cases = [
        {
            'name': '殺球 (Smash)',
            'params': {
                'overall_slope': 400.0,
                'highest_position_ratio': 0.05,
                'velocity': 600.0,
                'acceleration': 10.0,
                'y_range': 400.0,
                'high_ball_ratio': 0.1,
                'last_frames_low': True,
                'has_turning_point': False
            }
        },
        {
            'name': '高遠球 (Clear)',
            'params': {
                'overall_slope': -50.0,
                'highest_position_ratio': 0.9,
                'velocity': 300.0,
                'acceleration': -5.0,
                'y_range': 100.0,
                'high_ball_ratio': 0.95,
                'last_frames_low': False,
                'has_turning_point': False
            }
        },
        {
            'name': '切球 (Drop)',
            'params': {
                'overall_slope': 200.0,
                'highest_position_ratio': 0.1,
                'velocity': 250.0,
                'acceleration': -3.0,
                'y_range': 250.0,
                'high_ball_ratio': 0.2,
                'last_frames_low': True,
                'has_turning_point': False
            }
        }
    ]
    
    print("\n" + "=" * 60)
    print("測試預測")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\n📝 測試案例: {test_case['name']}")
        predicted_class, confidence, probabilities = classifier.predict(
            test_case['params'],
            return_probabilities=True
        )
        
        print(f"   預測結果: {predicted_class}")
        print(f"   信心度: {confidence:.2%}")
        print(f"   機率分布:")
        for label, prob in probabilities.items():
            print(f"      {label}: {prob:.2%}")
    
    print("\n✅ 測試完成！")


if __name__ == "__main__":
    test_inference()
