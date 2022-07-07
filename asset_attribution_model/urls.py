from django.urls import path

from .views import DoAssetAttribution

urlpatterns = [
    path('', DoAssetAttribution.as_view(), name='asset-attribution'),
]
