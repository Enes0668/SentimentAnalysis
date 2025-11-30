# analiz/tests.py
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.urls import reverse

from .models import EmotionRecord

User = get_user_model()

class EmotionRecordModelTests(TestCase):
    def test_create_emotion_record(self):
        user = User.objects.create_user(username="enes", password="test1234")
        rec = EmotionRecord.objects.create(
            user=user,
            emotion="happy",
            confidence=0.95,
        )
        self.assertEqual(rec.user, user)
        self.assertEqual(rec.emotion, "happy")
        self.assertAlmostEqual(rec.confidence, 0.95)

class EmotionStatsViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username="enes", password="test1234")

    def test_stats_view_requires_login(self):
        url = reverse("emotion_stats")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)  # login redirect

    def test_stats_view_shows_counts(self):
        EmotionRecord.objects.create(emotion="happy", confidence=0.9)
        EmotionRecord.objects.create(emotion="sad", confidence=0.6)

        self.client.login(username="enes", password="test1234")
        url = reverse("emotion_stats")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Toplam kayıt")
