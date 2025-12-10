"""
球種分類 - 軌跡記錄模組（改進版）

新的資料格式：
- 記錄完整 45 幀的 x, y 座標
- 明確標記哪些 frame 有資料、哪些缺失
- 讓神經網路自己學習特徵

資料結構範例：
{
    "timestamp": "2025-12-09T15:16:19.971572",
    "start_frame": 237,  # 開始偵測的 frame
    "end_frame": 282,    # 擊球判定的 frame
    "trajectory": [
        {"frame": 237, "x": 320.5, "y": 180.2, "detected": true},
        {"frame": 238, "x": null, "y": null, "detected": false},  # 缺幀
        {"frame": 239, "x": 325.1, "y": 185.3, "detected": true},
        ...
    ],
    "user_label": "Smash",
    "court_area": "left"  # 可選：球場區域
}
"""

import json
from datetime import datetime
from pathlib import Path
import numpy as np


class TrajectoryLogger:
    """記錄羽球軌跡的工具"""
    
    def __init__(self, log_file="./shot_trajectories.json"):
        self.log_file = Path(log_file)
        self.current_trajectory = []
        self.start_frame = None
        
    def add_point(self, frame_idx, x, y, detected=True):
        """
        新增一個軌跡點
        
        Args:
            frame_idx: 影格編號
            x: x 座標（如果 detected=False，可以是 None）
            y: y 座標（如果 detected=False，可以是 None）
            detected: 是否成功偵測到球
        """
        if self.start_frame is None:
            self.start_frame = frame_idx
        
        # 確保轉換為 Python 原生類型（避免 numpy 類型無法序列化）
        point = {
            "frame": int(frame_idx),
            "x": float(x) if (x is not None and detected) else None,
            "y": float(y) if (y is not None and detected) else None,
            "detected": bool(detected)
        }
        self.current_trajectory.append(point)
    
    def fill_missing_frames(self, end_frame):
        """
        填補缺失的幀（標記為未偵測）
        確保軌跡是連續的（即使有些幀沒有偵測到球）
        """
        if not self.current_trajectory or self.start_frame is None:
            return
        
        # 獲取已記錄的 frame 列表
        recorded_frames = {p['frame'] for p in self.current_trajectory}
        
        # 填補缺失的幀
        for frame in range(self.start_frame, end_frame + 1):
            if frame not in recorded_frames:
                self.current_trajectory.append({
                    "frame": frame,
                    "x": None,
                    "y": None,
                    "detected": False
                })
        
        # 按 frame 排序
        self.current_trajectory.sort(key=lambda p: p['frame'])
    
    def save_shot(self, end_frame, user_label, court_area=None, metadata=None):
        """
        儲存一次擊球的完整軌跡
        
        Args:
            end_frame: 擊球判定的 frame
            user_label: 使用者標註的球種（"Smash" / "Clear" / "Drop"）
            court_area: 球場區域（"left" / "right" / None）
            metadata: 其他額外資訊（字典）
        """
        if not self.current_trajectory:
            print("⚠️ 警告：軌跡為空，無法儲存")
            return None
        
        # 填補缺失的幀
        self.fill_missing_frames(end_frame)
        
        # 統計資訊
        detected_count = sum(1 for p in self.current_trajectory if p['detected'])
        total_frames = len(self.current_trajectory)
        
        # 建立記錄
        record = {
            "timestamp": datetime.now().isoformat(),
            "start_frame": int(self.start_frame),
            "end_frame": int(end_frame),
            "trajectory": self.current_trajectory,
            "user_label": user_label,
            "court_area": court_area,
            "stats": {
                "total_frames": total_frames,
                "detected_frames": detected_count,
                "detection_rate": detected_count / total_frames if total_frames > 0 else 0
            }
        }
        
        # 加入額外資訊（轉換 numpy 類型為 Python 原生類型）
        if metadata:
            record["metadata"] = self._convert_to_serializable(metadata)
        
        # 載入現有記錄
        history = self._load_history()
        history.append(record)
        
        # 儲存
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已儲存軌跡：{total_frames} 幀（偵測率 {detected_count}/{total_frames} = {record['stats']['detection_rate']:.1%}）")
        
        # 重置
        self.reset()
        
        return record
    
    def reset(self):
        """重置當前軌跡"""
        self.current_trajectory = []
        self.start_frame = None
    
    def _load_history(self):
        """載入歷史記錄"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def _convert_to_serializable(self, obj):
        """
        將 numpy 類型轉換為 Python 原生類型（遞迴處理）
        確保可以序列化為 JSON
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        elif obj is None:
            return None
        else:
            return obj
    
    def get_trajectory_length(self):
        """取得當前軌跡長度"""
        return len(self.current_trajectory)
    
    def get_detection_rate(self):
        """取得當前偵測率"""
        if not self.current_trajectory:
            return 0.0
        detected = sum(1 for p in self.current_trajectory if p['detected'])
        return detected / len(self.current_trajectory)


# ==================== 資料載入與轉換工具 ====================

def load_trajectories(log_file="./shot_trajectories.json"):
    """載入所有軌跡記錄"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def trajectory_to_array(trajectory, max_frames=45):
    """
    將軌跡轉換為固定長度的 numpy array
    
    Args:
        trajectory: 軌跡資料（list of dicts）
        max_frames: 最大幀數（預設 45）
    
    Returns:
        numpy array, shape=(max_frames, 3)
        每一行是 [x, y, detected_flag]
        缺失的幀用 [0, 0, 0] 填充
    """
    # 初始化 array
    arr = np.zeros((max_frames, 3), dtype=np.float32)
    
    for i, point in enumerate(trajectory[:max_frames]):  # 只取前 max_frames 幀
        if point['detected'] and point['x'] is not None and point['y'] is not None:
            arr[i, 0] = point['x']
            arr[i, 1] = point['y']
            arr[i, 2] = 1.0  # detected flag
        # else: 保持 [0, 0, 0]
    
    return arr


def convert_old_to_new_format(old_file="./shot_annotations.json", 
                                new_file="./shot_trajectories.json"):
    """
    將舊格式的標註檔案轉換為新格式
    （這是暫時的遷移工具，轉換後就不需要了）
    
    注意：舊格式沒有完整軌跡，只能創建假資料作為示範
    """
    print("⚠️ 警告：舊格式沒有完整軌跡資料，無法真正轉換")
    print("💡 建議：直接使用新系統重新標註")
    return None


# ==================== 使用範例 ====================

if __name__ == "__main__":
    # 建立 logger
    logger = TrajectoryLogger("./shot_trajectories.json")
    
    # 模擬記錄一次擊球軌跡
    print("📝 模擬記錄軌跡...")
    
    # 假設從 frame 100 開始偵測
    for frame in range(100, 145):
        if frame % 3 == 0:  # 模擬 1/3 的幀沒偵測到
            logger.add_point(frame, None, None, detected=False)
        else:
            # 模擬一個拋物線軌跡
            x = 320 + (frame - 100) * 5
            y = 200 + (frame - 120) ** 2 * 0.5
            logger.add_point(frame, x, y, detected=True)
    
    # 儲存
    logger.save_shot(
        end_frame=144,
        user_label="Smash",
        court_area="left",
        metadata={"note": "測試資料"}
    )
    
    # 載入並查看
    print("\n📊 載入軌跡資料...")
    trajectories = load_trajectories("./shot_trajectories.json")
    print(f"共有 {len(trajectories)} 筆記錄")
    
    if trajectories:
        # 轉換為 array
        traj_array = trajectory_to_array(trajectories[0]['trajectory'])
        print(f"\n軌跡 array shape: {traj_array.shape}")
        print(f"偵測幀數: {np.sum(traj_array[:, 2])}")
        print(f"前 5 幀資料:\n{traj_array[:5]}")
