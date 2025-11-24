from django.http import StreamingHttpResponse
from django.shortcuts import render
from .camera import VideoCamera

camera = VideoCamera()

def index(request):
    return render(request, "analiz/index.html")


def gen(camera):
    while True:
        frame = camera.get_jpeg_frame()
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

def video_feed(request):
    return StreamingHttpResponse(gen(camera),
        content_type="multipart/x-mixed-replace; boundary=frame")
