# translator\urls.py
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('asl/', views.index, name='asl'),
    path('about/', views.about, name='about'),
    path('learn-asl/', views.learn_asl, name='learn_asl'),
]