import os
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

class FaceHandler:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_embeddings = {}
        self.tracked_ids_seen = {}
        self.max_frames_to_process = 3
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(filename)[0]
                path = os.path.join(folder_path, filename)

                try:
                    embedding = DeepFace.represent(
                        img_path=path,
                        model_name="Facenet",
                        enforce_detection=False,
                        detector_backend="mtcnn"
                    )[0]["embedding"]
                    self.known_embeddings[name] = embedding
                    print(f"✅ Loaded face: {name}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")

    def identify_face(self, test_embedding, threshold=0.8, unknown_threshold=0.25):
        closest_person = "Unknown"
        closest_distance = float("inf")

        for name, known_embedding in self.known_embeddings.items():
            distance = cosine(test_embedding, known_embedding)
            if distance < threshold and distance < closest_distance:
                closest_distance = distance
                closest_person = name

        if closest_distance > unknown_threshold:
            closest_person = "Unknown"

        return closest_person, closest_distance

    def process_face(self, frame, bbox, tracking_id):
        if tracking_id not in self.tracked_ids_seen:
            self.tracked_ids_seen[tracking_id] = 0

        if self.tracked_ids_seen[tracking_id] >= self.max_frames_to_process:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"⚠️ Empty crop for ID {tracking_id}")
            return None

        try:
            faces = DeepFace.extract_faces(
                img_path=crop,
                detector_backend="mtcnn",
                enforce_detection=False,
                align=True
            )
        except:
            print(f"⚠️ No face detected in crop for ID {tracking_id}")
            return None

        if not faces:
            print(f"⚠️ No face for ID {tracking_id}")
            return None

        face_img = faces[0]["face"]

        try:
            embedding = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="skip"
            )[0]["embedding"]

            match_name, dist = self.identify_face(embedding)
            print(f"🧠 ID {tracking_id} match: {match_name} (distance: {dist:.4f})")

            self.tracked_ids_seen[tracking_id] += 1
            return match_name, dist

        except Exception as e:
            print(f"❌ DeepFace error for ID {tracking_id}: {e}")
            return None
