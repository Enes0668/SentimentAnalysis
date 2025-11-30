from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.db.models import Count
from analiz.models import EmotionRecord
from .camera import VideoCamera, camera_instance
from django.contrib.auth.decorators import login_required
import uuid
from django.utils import timezone
from django.db.models.functions import TruncDate

camera = VideoCamera()

def index(request):
    return render(request, "analiz/index.html")


def gen(camera):
    while True:
        frame = camera.get_jpeg_frame()
        if frame is None:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
        )

@login_required
def video_feed(request):
    global camera_instance

    session_id = request.session.get("emotion_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["emotion_session_id"] = session_id

    if camera_instance is None:
        camera_instance = VideoCamera(
            src=0,
            user=request.user,
            session_id=session_id
        )
    
    return StreamingHttpResponse(
        gen(camera_instance),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

@login_required
def emotion_stats(request):
    qs = EmotionRecord.objects.all()

    # Sadece ilgili kullanıcının verileri istenirse:
    user_only = request.GET.get("me") == "1"
    if user_only:
        qs = qs.filter(user=request.user)

    total_count = qs.count()

    # Duygu dağılımı
    emotion_counts = (
        qs.values('emotion')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    # Son 7 gün günlük dağılım
    seven_days_ago = timezone.now() - timezone.timedelta(days=7)
    daily_stats = (
        qs.filter(created_at__gte=seven_days_ago)
        .annotate(day=TruncDate('created_at'))
        .values('day', 'emotion')
        .annotate(count=Count('id'))
        .order_by('day', 'emotion')
    )

    context = {
        "total_count": total_count,
        "emotion_counts": emotion_counts,
        "daily_stats": daily_stats,
        "user_only": user_only,
    }
    return render(request, "analiz/emotion_stats.html", context)