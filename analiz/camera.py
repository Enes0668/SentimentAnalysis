import cv2
import threading
import numpy as np
import os
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import time

class VideoCamera:
    def __init__(self, src=0):
        # Kamera aç
        self.video = cv2.VideoCapture(src)
        if not self.video.isOpened():
            raise RuntimeError("Kamera açılamadı. Cihaz index'ini kontrol et.")

        # YOLOv8 yüz modeli
        self.face_model = YOLO("analiz/yolov8n-face.pt")

        # Duygu modeli (XCEPTION)
        model_path = os.path.join(os.path.dirname(__file__), "fer2013_mini_XCEPTION.102-0.66.hdf5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        self.emotion_model = load_model(model_path, compile=False)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Thread güvenliği
        self.lock = threading.Lock()
        self.running = True

        # Frame değişkenleri
        self.frame = None
        self.processed_frame = None

        # Threadler
        threading.Thread(target=self._update, daemon=True).start()
        threading.Thread(target=self._process, daemon=True).start()

    def __del__(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()

    # Video okuma thread'i
    def _update(self):
        while self.running:
            grabbed, frame = self.video.read()
            if not grabbed:
                continue
            with self.lock:
                self.frame = frame.copy()
            time.sleep(0.01)  # CPU yükünü azaltmak için küçük gecikme

    # Model tahmin thread'i
    def _process(self):
        while self.running:
            with self.lock:
                if self.frame is None:
                    continue
                frame = self.frame.copy()

            detections = self.face_model(frame)[0]
            for det in detections.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                face_input = self.preprocess_face(frame, (x1, y1, x2, y2))
                if face_input is not None:
                    try:
                        pred = self.emotion_model.predict(face_input, verbose=0)
                        emotion_label = self.emotions[np.argmax(pred)]
                    except:
                        emotion_label = "Error"
                else:
                    emotion_label = "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            with self.lock:
                self.processed_frame = frame

            time.sleep(0.01)  # CPU yükünü azaltmak için küçük gecikme

    # Yüzü XCEPTION için hazırlama
    def preprocess_face(self, frame, box):
        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        face_normalized = face_resized.astype('float32') / 255.0
        return np.expand_dims(np.expand_dims(face_normalized, -1), 0)

    # HTTP veya GUI için frame alma
    def get_jpeg_frame(self):
        with self.lock:
            if self.processed_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.processed_frame)
            return jpeg.tobytes()
