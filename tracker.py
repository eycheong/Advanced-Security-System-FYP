# tracker.py

class Tracker:
    def __init__(self, iou_threshold=0.5):
        self.next_id = 0
        self.tracks = []  # Each track: (id, box, class_name)
        self.iou_threshold = iou_threshold

    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = box1_area + box2_area - inter_area
        return inter_area / union if union else 0

    def track(self, detections, frame):
        updated_tracks = []
        tracking_results = []

        for det in detections:
            box = det["box"]
            class_name = det["class_name"]
            matched = False

            for track_id, track_box, track_class in self.tracks:
                if self.iou(box, track_box) > self.iou_threshold:
                    updated_tracks.append((track_id, box, class_name))
                    tracking_results.append((track_id, box, class_name))
                    matched = True
                    break

            if not matched:
                updated_tracks.append((self.next_id, box, class_name))
                tracking_results.append((self.next_id, box, class_name))
                self.next_id += 1

        self.tracks = updated_tracks
        return tracking_results  # Each: (id, box, class_name)
