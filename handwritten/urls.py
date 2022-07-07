from django.urls import path

from handwritten.service import HandwrittenAttribute

urlpatterns = [
    path('',HandwrittenAttribute.as_view(), name='handwritten-attribution'),
]
