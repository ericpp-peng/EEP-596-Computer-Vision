#!/bin/bash

# 羽球動作分析 - 一鍵執行腳本

echo "=========================================="
echo "羽球動作分析系統 - MVP"
echo "=========================================="
echo ""

# 移到專案目錄
cd "/Users/eric/Documents/UW 修課/2025 fall/EEP 596A CV/EEP-596-Computer-Vision/final-project"

# 檢查環境
echo "1️⃣ 檢查環境..."
python -c "from ultralytics import YOLO; import cv2; print('✅ 環境正常')" || exit 1
echo ""

# 選擇模式
echo "請選擇執行模式:"
echo "  1) 快速測試 (5分鐘, 建議先執行這個)"
echo "  2) 球場標定 (需要手動點擊)"
echo "  3) 完整分析 (需要先完成球場標定)"
echo ""
read -p "輸入選項 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "2️⃣ 執行快速測試..."
        python quick_test.py
        echo ""
        echo "✅ 完成! 輸出: quick_test_output.mp4"
        echo "執行: open quick_test_output.mp4"
        ;;
    2)
        echo ""
        echo "2️⃣ 啟動球場標定工具..."
        echo "請依序點擊球場四個角點:"
        echo "  左上 → 右上 → 右下 → 左下"
        echo "按 's' 儲存, 按 'q' 退出"
        echo ""
        python court_calibration.py
        ;;
    3)
        echo ""
        if [ ! -f "court_corners.pkl" ]; then
            echo "⚠️  找不到 court_corners.pkl"
            echo "請先執行選項 2 進行球場標定"
            exit 1
        fi
        echo "2️⃣ 執行完整分析..."
        python badminton_analysis.py
        echo ""
        echo "✅ 完成! 輸出: badminton_analysis_output.mp4"
        echo "執行: open badminton_analysis_output.mp4"
        ;;
    *)
        echo "無效選項"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "完成!"
echo "=========================================="
