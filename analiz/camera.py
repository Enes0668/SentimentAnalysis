import cv2
import threading
import time
import numpy as np
import os
from tensorflow.keras.models import load_model

class VideoCamera:
    def __init__(self, src=0):
        # 📷 Kamera aç
        self.video = cv2.VideoCapture(src)
        if not self.video.isOpened():
            raise RuntimeError("Kamera açılamadı. Cihaz index'ini kontrol et.")

        # 🧠 Mini-Xception FER2013 modeli yükle
        model_path = os.path.join(os.path.dirname(__file__), 'fer2013_mini_XCEPTION.102-0.66.hdf5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        self.emotion_model = load_model(model_path, compile=False)

        # 🎭 Duygu etiketleri
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # 🔍 Haarcascade yükle
        cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haarcascade dosyası bulunamadı: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # 🔁 Kamera thread başlat
        self.lock = threading.Lock()
        self.grabbed, self.frame = self.video.read()
        self.running = True
        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        while self.running:
            grabbed, frame = self.video.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def preprocess_face(self, frame, face_rect, target_size=(64, 64)):
        (x, y, w, h) = face_rect
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            return None

        # Gri tonlama (Mini-Xception gri giriş bekliyor)
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(face_gray, target_size)
        normalized = resized.astype('float32') / 255.0
        # (1, 64, 64, 1)
        return np.expand_dims(np.expand_dims(normalized, -1), 0)

    def get_frame_with_faces_and_emotions(self):
        frame = self.get_frame()
        if frame is None:
            return None, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        results = []
        for (x, y, w, h) in faces:
            face_input = self.preprocess_face(frame, (x, y, w, h))
            if face_input is not None:
                try:
                    prediction = self.emotion_model.predict(face_input, verbose=0)
                    emotion_index = np.argmax(prediction)
                    emotion_label = self.emotions[emotion_index]
                except Exception as e:
                    print("Tahmin hatası:", e)
                    emotion_label = "Error"
            else:
                emotion_label = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            results.append({'rect': (x, y, w, h), 'emotion': emotion_label})

        return frame, results

    def __del__(self):
        self.running = False
        try:
            self.video.release()
        except:
            pass

