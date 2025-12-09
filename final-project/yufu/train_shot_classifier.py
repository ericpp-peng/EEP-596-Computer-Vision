"""
訓練球種分類器
使用標註資料訓練神經網路來分類羽球擊球類型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from shot_classifier_model import ShotClassifier, LightShotClassifier
from shot_dataset import load_data, create_dataloaders, save_scaler, LABEL_NAMES


class EarlyStopping:
    """Early stopping 機制"""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def calculate_class_weights(label_counts, num_classes=3):
    """
    計算類別權重（處理不平衡資料）
    
    Args:
        label_counts: dict of {class_idx: count}
        num_classes: 總類別數
    
    Returns:
        weights: tensor of shape (num_classes,)
    """
    total_samples = sum(label_counts.values())
    weights = []
    
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # 避免除以零
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        # 前向傳播
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        # 統計
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels


def plot_confusion_matrix(y_true, y_pred, save_path='./confusion_matrix.png'):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 混淆矩陣已儲存至: {save_path}")
    plt.close()


def plot_training_history(history, save_path='./training_history.png'):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    if 'val_acc' in history and len(history['val_acc']) > 0:
        ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📈 訓練歷史已儲存至: {save_path}")
    plt.close()


def train(
    model,
    train_loader,
    test_loader,
    device,
    num_epochs=100,
    learning_rate=0.001,
    class_weights=None,
    patience=15,
    save_dir='./shot_classifier_weights'
):
    """
    訓練球種分類器
    
    Args:
        model: 模型
        train_loader: 訓練集 DataLoader
        test_loader: 測試集 DataLoader
        device: 'cpu', 'cuda', or 'mps'
        num_epochs: 訓練輪數
        learning_rate: 學習率
        class_weights: 類別權重（處理不平衡資料）
        patience: Early stopping 的耐心值
        save_dir: 儲存權重的目錄
    """
    model = model.to(device)
    
    # 損失函數（加入類別權重）
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 學習率調度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # 訓練歷史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 最佳模型追蹤
    best_val_acc = 0
    best_epoch = 0
    
    # 建立儲存目錄
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("開始訓練球種分類器")
    print("=" * 60)
    print(f"🖥️  裝置: {device}")
    print(f"📦 訓練集大小: {len(train_loader.dataset)}")
    if test_loader:
        print(f"📦 測試集大小: {len(test_loader.dataset)}")
    print(f"🔢 Epochs: {num_epochs}")
    print(f"📊 Learning Rate: {learning_rate}")
    print(f"⚖️  Class Weights: {class_weights}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 訓練
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 評估
        if test_loader:
            val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # 學習率調度
            scheduler.step(val_loss)
            
            # 儲存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_path = os.path.join(save_dir, f'best_model_{timestamp}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, best_model_path)
            
            # Early stopping
            early_stopping(val_loss)
            
            # 輸出訓練資訊
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if early_stopping.early_stop:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        else:
            # 沒有測試集的情況
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    
    # 儲存最終模型
    final_model_path = os.path.join(save_dir, f'final_model_{timestamp}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
    }, final_model_path)
    
    print("\n" + "=" * 60)
    print("✅ 訓練完成！")
    print("=" * 60)
    if test_loader:
        print(f"🏆 最佳驗證準確率: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
        print(f"💾 最佳模型: {best_model_path}")
    print(f"💾 最終模型: {final_model_path}")
    
    return history, best_model_path if test_loader else final_model_path


def main():
    # ==================== 參數設定 ====================
    ANNOTATION_FILE = './shot_annotations.json'
    SAVE_DIR = './shot_classifier_weights'
    
    # 訓練參數
    NUM_EPOCHS = 50  # 小數據集不需要太多輪
    LEARNING_RATE = 0.001
    BATCH_SIZE = 4  # 小數據集用小 batch
    TEST_SIZE = 0.2
    PATIENCE = 20
    USE_LIGHT_MODEL = True  # 使用輕量模型（適合小數據集）
    
    # ==================== 載入資料 ====================
    X_train, X_test, y_train, y_test, scaler, label_counts = load_data(
        json_path=ANNOTATION_FILE,
        test_size=TEST_SIZE,
        random_state=42
    )
    
    # 儲存 scaler
    save_scaler(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
    
    # 建立 DataLoader
    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    # ==================== 建立模型 ====================
    # 設定裝置
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ 使用 MPS (Metal) GPU 加速")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("✅ 使用 CUDA GPU 加速")
    else:
        device = 'cpu'
        print("⚠️  使用 CPU")
    
    # 建立模型
    if USE_LIGHT_MODEL:
        model = LightShotClassifier(input_dim=8, num_classes=3, dropout=0.2)
        print("🔹 使用輕量級模型")
    else:
        model = ShotClassifier(input_dim=8, hidden_dims=[64, 32, 16], num_classes=3, dropout=0.3)
        print("🔹 使用標準模型")
    
    # 計算類別權重
    class_weights = calculate_class_weights(label_counts, num_classes=3)
    print(f"⚖️  類別權重: {class_weights}")
    
    # ==================== 訓練 ====================
    history, best_model_path = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        patience=PATIENCE,
        save_dir=SAVE_DIR
    )
    
    # ==================== 評估 ====================
    if test_loader:
        # 載入最佳模型
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # 評估
        criterion = nn.CrossEntropyLoss()
        _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
        
        print("\n" + "=" * 60)
        print("📊 測試集評估結果")
        print("=" * 60)
        print(f"準確率: {test_acc:.2f}%\n")
        
        # 分類報告
        print("分類報告:")
        print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
        
        # 繪製混淆矩陣
        plot_confusion_matrix(y_true, y_pred, 
                             save_path=os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    
    # ==================== 繪製訓練歷史 ====================
    plot_training_history(history, 
                         save_path=os.path.join(SAVE_DIR, 'training_history.png'))
    
    # ==================== 儲存訓練資訊 ====================
    info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'model_type': 'LightShotClassifier' if USE_LIGHT_MODEL else 'ShotClassifier',
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'label_counts': {LABEL_NAMES[k]: int(v) for k, v in label_counts.items()},
        'final_train_acc': float(history['train_acc'][-1]),
        'final_train_loss': float(history['train_loss'][-1]),
    }
    
    if test_loader:
        info['best_val_acc'] = float(checkpoint['val_acc'])
        info['best_val_loss'] = float(checkpoint['val_loss'])
        info['test_acc'] = float(test_acc)
    
    info_path = os.path.join(SAVE_DIR, 'training_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 訓練資訊已儲存至: {info_path}")
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
