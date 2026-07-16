"""URL patterns for tableocr app."""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_file, name='upload'),
    path('download/csv/<str:session_id>/', views.download_csv, name='download_csv'),
    path('download/image/<str:session_id>/', views.download_image, name='download_image'),
    path('preview/csv/<str:session_id>/', views.preview_csv, name='preview_csv'),
    path('preview/image/<str:session_id>/', views.preview_image, name='preview_image'),
]
