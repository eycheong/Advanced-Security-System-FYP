# yolo_detector.py

import torch

class YoloDetector:
    def __init__(self, model_path: str, confidence: float = 0.25):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = confidence
        self.class_names = self.model.names  # Class index to name mapping

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls)
            class_name = self.class_names[class_id]
            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": class_id,
                "class_name": class_name
            })
        return detections
