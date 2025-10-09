# analiz/camera.py
import cv2
import threading
import time
import numpy as np
import os

class VideoCamera:
    def __init__(self, src=0):
        self.video = cv2.VideoCapture(src)
        if not self.video.isOpened():
            raise RuntimeError("Kamera açılamadı. Doğru cihaz index'ini (0,1,...) kontrol et veya WSL kullanıyorsan yerel cihaz yoktur.")
        
        # --- 🔹 Haarcascade dosyasını doğru bulmak için tam yol tanımla ---
        cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haarcascade dosyası bulunamadı: {cascade_path}")
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Yüz tanıma modeli yüklenemedi! XML dosyası bozuk veya okunamıyor.")
        # -------------------------------------------------------------

        self.lock = threading.Lock()
        self.grabbed, self.frame = self.video.read()
        self.running = True
        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        # Kameradan sürekli okuyup son frame'i sakla
        while self.running:
            grabbed, frame = self.video.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)

    def get_frame(self):
        # Kopyalanmış BGR frame döndür
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_frame_with_faces(self):
        # Frame üstünde yüzleri tespit edip rectangle çizilmiş frame döndürür
        frame = self.get_frame()
        if frame is None:
            return None, []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame, faces

    def get_face_crop_preprocessed(self, face_rect, target_size=(48,48)):
        frame = self.get_frame()
        if frame is None:
            return None
        (x, y, w, h) = face_rect
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size)
        normalized = resized.astype('float32') / 255.0
        return np.expand_dims(normalized, axis=-1)

    def __del__(self):
        self.running = False
        try:
            self.video.release()
        except:
            pass
