import cv2
import threading
import numpy as np
import os
import time

from tensorflow.keras.models import load_model
from ultralytics import YOLO
from django.utils import timezone
from test_app.models import EmotionRecord

camera_instance = None


class VideoCamera:
    def __init__(self, src=0, user=None, session_id=None):
        self.video = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)

        if not self.video.isOpened():
            raise RuntimeError("Kamera açılamadı")

        self.frame = None
        self.processed_frame = self._empty_frame()
        self.running = True

        self.user = user
        self.session_id = session_id or f"session-{int(time.time())}"

        self.last_save_time = 0
        self.save_interval = 2.0

        self.boxes = []
        self.max_faces = 2

        self.frame_count = 0
        self.yolo_interval = 8
        self.emotion_interval = 6

        self.emotions = [
            "angry", "disgust", "fear",
            "happy", "sad", "surprise", "neutral"
        ]

        threading.Thread(target=self._update, daemon=True).start()
        threading.Thread(target=self._load_models, daemon=True).start()

    def __del__(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()

    def _empty_frame(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "Loading camera...",
            (180, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        return img

    def _update(self):
        while self.running:
            ret, frame = self.video.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.01)

    def _load_models(self):
        self.face_model = YOLO("analiz/yolov8n-face.pt")
        model_path = os.path.join(os.path.dirname(__file__), "emotion_model_tf12.h5")
        self.emotion_model = load_model(model_path, compile=False)

        dummy = np.zeros((1, 197, 197, 3), dtype=np.float32)
        self.emotion_model.predict(dummy, verbose=0)

        while self.frame is None:
            time.sleep(0.05)

        _ = self.face_model(self.frame, imgsz=320, verbose=False)

        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        last_emotions = {}

        while self.running:
            if self.frame is None:
                time.sleep(0.01)
                continue

            frame = self.frame.copy()
            self.frame_count += 1
            now = time.time()

            if self.frame_count % self.yolo_interval == 0 or not self.boxes:
                detections = self.face_model(frame, imgsz=320, verbose=False)[0]
                boxes = []

                for det in detections.boxes:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

                boxes.sort(
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    reverse=True
                )
                self.boxes = boxes[:self.max_faces]

            for i, (x1, y1, x2, y2) in enumerate(self.boxes):
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                emotion, conf = last_emotions.get(i, ("...", 0.0))

                if self.frame_count % self.emotion_interval == 0:
                    face_in = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_in = cv2.resize(face_in, (197, 197))
                    face_in = face_in.astype("float32") / 255.0
                    face_in = np.expand_dims(face_in, axis=0)

                    pred = self.emotion_model.predict(face_in, verbose=0)
                    idx = int(np.argmax(pred))
                    emotion = self.emotions[idx]
                    conf = float(np.max(pred))
                    last_emotions[i] = (emotion, conf)

                    if now - self.last_save_time > self.save_interval:
                        self._save_emotion(emotion, conf)
                        self.last_save_time = now

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            self.processed_frame = frame

    def _save_emotion(self, emotion, confidence):
        try:
            EmotionRecord.objects.create(
                user=self.user if getattr(self.user, "is_authenticated", False) else None,
                emotion=emotion,
                confidence=confidence,
                session_id=self.session_id,
                created_at=timezone.now()
            )
        except:
            pass

    def get_jpeg_frame(self):
        _, jpeg = cv2.imencode(".jpg", self.processed_frame)
        return jpeg.tobytes()
