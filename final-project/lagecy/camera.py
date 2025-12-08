import cv2

print("Testing OpenCV camera indexes...\n")

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Index {i} → OPENED (ret={ret})")
    else:
        print(f"Index {i} → NOT AVAILABLE")
    cap.release()
