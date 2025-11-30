# analiz/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('istatistikler/', views.emotion_stats, name='emotion_stats'),
]
