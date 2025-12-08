import torch
from ultralytics.nn.tasks import PoseModel

with torch.serialization.safe_globals([PoseModel]):
    from ultralytics import YOLO
    model = YOLO("yolov8n-pose.pt")

print("loaded OK!")
