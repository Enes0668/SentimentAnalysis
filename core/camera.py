import cv2
import threading
import numpy as np
import os
import time
from collections import deque

from tensorflow.keras.models import load_model
from ultralytics import YOLO
from django.utils import timezone

from test_app.models import EmotionRecord

camera_instance = None

class VideoCamera:
    def __init__(self, src=0, user=None, session_id=None):
        self.video = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.video.isOpened():
            raise RuntimeError("Kamera açılamadı.")

        self.face_model = YOLO("analiz/yolov8n-face.pt")

        model_path = os.path.join(
            os.path.dirname(__file__),
            "fer2013_mini_XCEPTION.102-0.66.hdf5"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")

        self.emotion_model = load_model(model_path, compile=False)
        self.emotions = [
            'angry', 'disgust', 'fear',
            'happy', 'sad', 'surprise', 'neutral'
        ]

        self.lock = threading.Lock()
        self.running = True
        self.frame = None
        self.processed_frame = None

        self.user = user
        self.session_id = session_id or f"session-{int(time.time())}"

        self.last_save_time = 0
        self.save_interval = 2.0

        self.tracks = {}
        self.next_track_id = 1
        self.track_max_age = 0.8
        self.match_dist_thresh = 80.0
        self.max_faces = 10

        threading.Thread(target=self._update, daemon=True).start()
        threading.Thread(target=self._process, daemon=True).start()

    def __del__(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()


    def _update(self):
        while self.running:
            grabbed, frame = self.video.read()
            if grabbed:
                with self.lock:
                    self.frame = frame.copy()
            time.sleep(0.01)


    def _process(self):
        while self.running:
            with self.lock:
                if self.frame is None:
                    continue
                frame = self.frame.copy()

            detections = self.face_model(frame)[0]

            boxes = []
            for det in detections.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                boxes.append((x1, y1, x2, y2))

            boxes.sort(
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                reverse=True
            )
            boxes = boxes[:self.max_faces]

            now_ts = time.time()
            self._cleanup_tracks(now_ts)

            for bbox in boxes:
                track_id = self._assign_track(bbox, now_ts)
                x1, y1, x2, y2 = self._smooth_bbox_for_track(track_id, bbox)

                face_input = self.preprocess_face(frame, (x1, y1, x2, y2))
                if face_input is None:
                    continue

                pred = self.emotion_model.predict(face_input, verbose=0)
                idx = int(np.argmax(pred))
                emotion_label = self.emotions[idx]
                confidence = float(np.max(pred))

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}: {emotion_label} ({confidence:.2f})",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                if now_ts - self.last_save_time > self.save_interval:
                    self._save_emotion(emotion_label, confidence)
                    self.last_save_time = now_ts

            with self.lock:
                self.processed_frame = frame

            time.sleep(0.01)


    def _bbox_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _cleanup_tracks(self, now_ts):
        to_delete = []
        for tid, info in self.tracks.items():
            if now_ts - info["last_seen"] > self.track_max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

    def _assign_track(self, bbox, now_ts):
        cx, cy = self._bbox_centroid(bbox)

        best_id = None
        best_dist = float("inf")

        for tid, info in self.tracks.items():
            pcx, pcy = info["centroid"]
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is None or best_dist > self.match_dist_thresh:
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = {
                "history": deque(maxlen=5),
                "centroid": (cx, cy),
                "last_seen": now_ts
            }
            return tid

        self.tracks[best_id]["centroid"] = (cx, cy)
        self.tracks[best_id]["last_seen"] = now_ts
        return best_id

    def _smooth_bbox_for_track(self, track_id, bbox):
        info = self.tracks.get(track_id)
        if info is None:
            return bbox
        info["history"].append(bbox)
        return tuple(np.mean(info["history"], axis=0).astype(int))


    def preprocess_face(self, frame, box):
        h, w, _ = frame.shape
        x1, y1, x2, y2 = box

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        try:
            face_resized = cv2.resize(gray, (64, 64))
        except Exception:
            return None

        face_norm = face_resized.astype("float32") / 255.0
        return np.expand_dims(np.expand_dims(face_norm, -1), 0)


    def _save_emotion(self, emotion_label, confidence):
        try:
            EmotionRecord.objects.create(
                user=self.user if getattr(self.user, "is_authenticated", False) else None,
                emotion=emotion_label,
                confidence=confidence,
                session_id=self.session_id,
                created_at=timezone.now()
            )
        except Exception as e:
            print("Emotion kayıt hatası:", e)


    def get_jpeg_frame(self):
        with self.lock:
            if self.processed_frame is None:
                return None
            _, jpeg = cv2.imencode(".jpg", self.processed_frame)
            return jpeg.tobytes()
