"""
球種分類器模型定義
使用軌跡參數來分類羽球擊球類型（Smash/Clear/Drop）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShotClassifier(nn.Module):
    """
    球種分類神經網路
    
    輸入特徵：
    - overall_slope: 整體斜率
    - highest_position_ratio: 最高點位置比例
    - velocity: 速度
    - acceleration: 加速度
    - y_range: 垂直範圍
    - high_ball_ratio: 高處停留比例
    - last_frames_low: 最後幾幀是否在低處 (0/1)
    - has_turning_point: 是否有轉折點 (0/1)
    
    輸出：3 個類別（Smash, Clear, Drop）的機率
    """
    
    def __init__(self, input_dim=8, hidden_dims=[64, 32, 16], num_classes=3, dropout=0.3):
        """
        Args:
            input_dim: 輸入特徵數量
            hidden_dims: 隱藏層維度列表
            num_classes: 分類類別數量
            dropout: Dropout 比例
        """
        super(ShotClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 建立隱藏層
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 輸出層
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """使用 Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 訓練時若 batch size = 1，BatchNorm 會出錯，需要特殊處理
        if self.training and x.size(0) == 1:
            self.eval()
            result = self.network(x)
            self.train()
            return result
        return self.network(x)
    
    def predict(self, x):
        """
        預測類別
        
        Args:
            x: (batch_size, input_dim) 的特徵張量
        
        Returns:
            predictions: (batch_size,) 的預測類別索引
            probabilities: (batch_size, num_classes) 的機率分布
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


class LightShotClassifier(nn.Module):
    """
    輕量級球種分類器（適合小數據集）
    """
    
    def __init__(self, input_dim=8, num_classes=3, dropout=0.2):
        super(LightShotClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(16, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 訓練時若 batch size = 1，BatchNorm 會出錯，需要特殊處理
        if self.training and x.size(0) == 1:
            self.eval()
            x_out = self.fc1(x)
            x_out = F.relu(x_out)
            x_out = self.dropout1(x_out)
            
            x_out = self.fc2(x_out)
            x_out = F.relu(x_out)
            x_out = self.dropout2(x_out)
            
            x_out = self.fc3(x_out)
            self.train()
            return x_out
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        """預測類別"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


if __name__ == "__main__":
    # 測試模型
    print("=" * 60)
    print("測試球種分類器模型")
    print("=" * 60)
    
    # 創建模型
    model = ShotClassifier(input_dim=8, hidden_dims=[64, 32, 16], num_classes=3)
    light_model = LightShotClassifier(input_dim=8, num_classes=3)
    
    # 統計參數數量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n標準模型參數數量: {count_parameters(model):,}")
    print(f"輕量模型參數數量: {count_parameters(light_model):,}")
    
    # 測試前向傳播
    batch_size = 4
    input_dim = 8
    x = torch.randn(batch_size, input_dim)
    
    print(f"\n輸入形狀: {x.shape}")
    
    # 標準模型
    output = model(x)
    print(f"標準模型輸出形狀: {output.shape}")
    
    predictions, probabilities = model.predict(x)
    print(f"預測類別: {predictions}")
    print(f"機率分布:\n{probabilities}")
    
    # 輕量模型
    output_light = light_model(x)
    print(f"\n輕量模型輸出形狀: {output_light.shape}")
    
    predictions_light, probabilities_light = light_model.predict(x)
    print(f"預測類別: {predictions_light}")
    print(f"機率分布:\n{probabilities_light}")
    
    print("\n✅ 模型測試完成！")
