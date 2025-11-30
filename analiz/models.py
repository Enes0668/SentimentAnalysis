# analiz/models.py
from django.conf import settings
from django.db import models

EMOTIONS = [
    ('angry', 'Angry'),
    ('disgust', 'Disgust'),
    ('fear', 'Fear'),
    ('happy', 'Happy'),
    ('sad', 'Sad'),
    ('surprise', 'Surprise'),
    ('neutral', 'Neutral'),
]

class EmotionRecord(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='emotion_records'
    )
    emotion = models.CharField(max_length=20, choices=EMOTIONS)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"{self.user or 'Anon'} - {self.emotion} - {self.created_at:%Y-%m-%d %H:%M}"
