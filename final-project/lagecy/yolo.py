from ultralytics import YOLO
import cv2
import numpy as np
import time

det_model = YOLO("yolov8n.pt", verbose=False)
pose_model = YOLO("yolov8n-pose.pt", verbose=False)

cap = cv2.VideoCapture("20250711_short.mp4")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output_pose.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

start_time = time.time()
last_report = start_time
frame_id = 0

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLO detection ---
    det_results = det_model(frame, imgsz=320, verbose=False)[0]

    h, w, _ = frame.shape
    center = np.array([w/2, h/2])

    best_score = -1
    best_box = None

    # select main player
    for box in det_results.boxes:
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2-x1)*(y2-y1)
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        dist = np.linalg.norm(np.array([cx,cy]) - center)

        score = area - dist * 50
        if score > best_score:
            best_score = score
            best_box = (x1,y1,x2,y2)

    if best_box is not None:
        x1,y1,x2,y2 = best_box
        crop = frame[y1:y2, x1:x2]

        pose_results = pose_model(crop, imgsz=320, verbose=False)[0]

        if len(pose_results.keypoints) > 0:
            kpts = pose_results.keypoints.xy[0].cpu().numpy()

            # draw keypoints back on original frame
            for (px, py) in kpts:
                px = int(px + x1)
                py = int(py + y1)
                cv2.circle(frame, (px, py), 3, (0,255,0), -1)

    # write output video
    out.write(frame)
    frame_id += 1

    # progress every 30 seconds
    now = time.time()
    if now - last_report >= 30:
        percent = frame_id / total_frames * 100
        elapsed = format_time(now - start_time)
        print(f"Progress: {percent:.2f}% ({frame_id}/{total_frames}) | Elapsed: {elapsed}")
        last_report = now

cap.release()
out.release()

print("🎉 Done! 已輸出 output_pose.mp4")
