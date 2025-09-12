from django.urls import path
from .views import (
    FingerPrintRegisterView, 
    FingerPrintDetectView, 
    FingerPrintListView, 
    FingerPrintDetailView,
    FingerPrintRegenerateEmbeddingsView
)

urlpatterns = [
    path('fingerprint/register/', FingerPrintRegisterView.as_view(), name='fingerprint-register'),
    path('fingerprint/detect/', FingerPrintDetectView.as_view(), name='fingerprint-detect'),
    path('fingerprint/list/', FingerPrintListView.as_view(), name='fingerprint-list'),
    path('fingerprint/<int:fingerprint_id>/', FingerPrintDetailView.as_view(), name='fingerprint-detail'),
    path('fingerprint/regenerate-embeddings/', FingerPrintRegenerateEmbeddingsView.as_view(), name='fingerprint-regenerate-embeddings'),
]
