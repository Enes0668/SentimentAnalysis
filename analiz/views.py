# analiz/views.py
from django.shortcuts import render
from django.http import StreamingHttpResponse
from .camera import VideoCamera
import cv2

camera = VideoCamera()  # uygulama import edildiğinde kamera başlatılır (dev amaçlı uygun)

def index(request):
    return render(request, 'analiz/index.html')

def frame_generator():
    while True:
        frame, faces = camera.get_frame_with_faces()
        if frame is None:
            continue
        # JPEG'e çevir
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(frame_generator(), content_type='multipart/x-mixed-replace; boundary=frame')
