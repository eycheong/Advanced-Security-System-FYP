import cv2
from yolo_detector import YoloDetector
from tracker import Tracker
from face_handler import FaceHandler
from email_alert import send_email_alert

# Paths to the model and video
MODEL_PATH = r"C:\Users\Win 11\yolov5\runs\train\exp10\weights\best.pt"
#VIDEO_PATH = r"C:\Users\Win 11\Desktop\face testinh.mp4"
#VIDEO_PATH = r"C:\Users\Win 11\Desktop\face_footage_2q.mp4"
VIDEO_PATH = r"C:\Users\Win 11\Downloads\ey.MP4"

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
    tracker = Tracker()
    face_handler = FaceHandler(known_faces_dir=r"C:\Users\Win 11\Youtube_Tracker - Copy\known_faces")

    cap = cv2.VideoCapture(VIDEO_PATH)
    #cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracking_results = tracker.track(detections, frame)

        for tracking_id, bbox, class_name in tracking_results:
            x1, y1, x2, y2 = map(int, bbox)
            label = f"{class_name} ID:{tracking_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if class_name == "face":
                result = face_handler.process_face(frame, bbox, tracking_id)
                if result is not None:
                    match_name, dist = result

                    if match_name == "Unknown":
                        subject = f"Unknown Face Detected (ID: {tracking_id})"
                        body = f"An unknown face was detected with ID {tracking_id}. Distance: {dist:.4f}. Please verify."
                        send_email_alert(subject, body, "cheong.eeying@icloud.com")
                    elif dist < 0.8:
                        print(f"Known face detected: {match_name} (ID: {tracking_id}), Distance: {dist:.4f}")


        frame_resized = cv2.resize(frame, (640, 640))
        cv2.imshow("Tracking", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
