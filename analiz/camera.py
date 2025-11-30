# analiz/camera.py
import cv2
import threading
import numpy as np
import os
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import time

from django.utils import timezone
from .models import EmotionRecord  # <-- yeni
camera_instance = None

class VideoCamera:
    def __init__(self, src=0, user=None, session_id=None):
        self.video = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.video.isOpened():
            raise RuntimeError("Kamera açılamadı. Cihaz index'ini kontrol et.")

        self.face_model = YOLO("analiz/yolov8n-face.pt")

        model_path = os.path.join(os.path.dirname(__file__), "fer2013_mini_XCEPTION.102-0.66.hdf5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        self.emotion_model = load_model(model_path, compile=False)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.lock = threading.Lock()
        self.running = True

        self.frame = None
        self.processed_frame = None

        self.user = user
        self.session_id = session_id or f"session-{int(time.time())}"
        self.last_emotion = None
        self.last_save_time = 0
        self.save_interval = 2.0 

        threading.Thread(target=self._update, daemon=True).start()
        threading.Thread(target=self._process, daemon=True).start()

    def __del__(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()

    def _update(self):
        while self.running:
            grabbed, frame = self.video.read()
            if not grabbed:
                continue
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
            current_emotion = None
            current_confidence = None

            for det in detections.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                face_input = self.preprocess_face(frame, (x1, y1, x2, y2))
                if face_input is not None:
                    try:
                        pred = self.emotion_model.predict(face_input, verbose=0)
                        emotion_index = int(np.argmax(pred))
                        emotion_label = self.emotions[emotion_index]
                        confidence = float(np.max(pred))
                    except Exception:
                        emotion_label = "Error"
                        confidence = None
                else:
                    emotion_label = "Unknown"
                    confidence = None

                current_emotion = emotion_label
                current_confidence = confidence

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if current_emotion and current_emotion not in ["Unknown", "Error"]:
                now_ts = time.time()
                if (
                    (self.last_emotion != current_emotion)
                    or (now_ts - self.last_save_time) > self.save_interval
                ):
                    self._save_emotion(current_emotion, current_confidence)
                    self.last_emotion = current_emotion
                    self.last_save_time = now_ts

            with self.lock:
                self.processed_frame = frame

            time.sleep(0.01)

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
            # İstersen loglayabilirsin:
            print("Emotion kayıt hatası:", e)

    def preprocess_face(self, frame, box):
        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        face_normalized = face_resized.astype('float32') / 255.0
        return np.expand_dims(np.expand_dims(face_normalized, -1), 0)

    def get_jpeg_frame(self):
        with self.lock:
            if self.processed_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.processed_frame)
            return jpeg.tobytes()
