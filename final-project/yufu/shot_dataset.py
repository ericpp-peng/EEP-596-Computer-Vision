"""
球種分類資料集處理
從 shot_annotations.json 讀取標註資料並建立 PyTorch Dataset
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# 類別映射
LABEL_MAP = {
    'Smash': 0,
    'Clear': 1,
    'Drop': 2
}

LABEL_NAMES = ['Smash', 'Clear', 'Drop']


class ShotDataset(Dataset):
    """球種分類資料集"""
    
    def __init__(self, features, labels):
        """
        Args:
            features: (N, 8) numpy array of features
            labels: (N,) numpy array of class indices
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_features(sample):
    """
    從標註樣本中提取特徵
    
    Args:
        sample: dict from shot_annotations.json
    
    Returns:
        features: list of 8 features
    """
    params = sample['parameters']
    
    # 提取數值特徵
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
    
    return features


def load_data(json_path='./shot_annotations.json', test_size=0.2, random_state=42):
    """
    載入標註資料並分割訓練/測試集
    
    Args:
        json_path: 標註檔案路徑
        test_size: 測試集比例
        random_state: 隨機種子
    
    Returns:
        X_train, X_test, y_train, y_test: 特徵和標籤
        scaler: StandardScaler 物件（用於測試時的標準化）
        label_counts: 各類別樣本數
    """
    print(f"📂 載入資料：{json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   總樣本數: {len(data)}")
    
    # 過濾出有 user_label 的樣本
    labeled_data = [s for s in data if 'user_label' in s and s['user_label']]
    print(f"   已標註樣本: {len(labeled_data)}")
    
    # 提取特徵和標籤
    features_list = []
    labels_list = []
    
    for sample in labeled_data:
        label = sample['user_label']
        if label not in LABEL_MAP:
            print(f"   ⚠️  跳過未知標籤: {label}")
            continue
        
        features = extract_features(sample)
        features_list.append(features)
        labels_list.append(LABEL_MAP[label])
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    
    # 統計各類別數量
    unique, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique, counts))
    
    print(f"\n📊 類別分布:")
    for label_idx, count in label_counts.items():
        print(f"   {LABEL_NAMES[label_idx]}: {count} 樣本")
    
    # 分割訓練/測試集（分層抽樣）
    if len(X) < 10:
        print(f"\n⚠️  樣本數太少 ({len(X)})，建議至少 20 個樣本")
        print(f"   使用全部資料進行訓練，無測試集")
        X_train, X_test = X, np.array([])
        y_train, y_test = y, np.array([])
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    
    print(f"\n✂️  資料分割:")
    print(f"   訓練集: {len(X_train)} 樣本")
    print(f"   測試集: {len(X_test)} 樣本")
    
    # 特徵標準化（使用訓練集計算均值和標準差）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if len(X_test) > 0:
        X_test = scaler.transform(X_test)
    
    print(f"\n📏 特徵標準化完成")
    print(f"   特徵均值: {scaler.mean_}")
    print(f"   特徵標準差: {scaler.scale_}")
    
    return X_train, X_test, y_train, y_test, scaler, label_counts


def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=8, num_workers=0):
    """
    建立 DataLoader
    
    Args:
        X_train, X_test, y_train, y_test: 特徵和標籤
        batch_size: batch 大小
        num_workers: 資料載入的 worker 數量
    
    Returns:
        train_loader, test_loader: DataLoader 物件
    """
    train_dataset = ShotDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    if len(X_test) > 0:
        test_dataset = ShotDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    else:
        test_loader = None
    
    return train_loader, test_loader


def save_scaler(scaler, path='./scaler.pkl'):
    """儲存 Scaler（用於推理時的特徵標準化）"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler 已儲存至: {path}")


def load_scaler(path='./scaler.pkl'):
    """載入 Scaler"""
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


if __name__ == "__main__":
    print("=" * 60)
    print("測試球種分類資料集")
    print("=" * 60)
    
    # 載入資料
    X_train, X_test, y_train, y_test, scaler, label_counts = load_data(
        json_path='./shot_annotations.json',
        test_size=0.2,
        random_state=42
    )
    
    # 建立 DataLoader
    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test,
        batch_size=4
    )
    
    # 測試 DataLoader
    print(f"\n🔄 測試 DataLoader:")
    for features, labels in train_loader:
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   First batch features:\n{features}")
        print(f"   First batch labels: {labels}")
        break
    
    # 儲存 scaler
    save_scaler(scaler, './scaler.pkl')
    
    print("\n✅ 資料集測試完成！")
