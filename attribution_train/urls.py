from django.urls import path

from .views import OnlineAttributionTrain

urlpatterns = [
    path('', OnlineAttributionTrain.as_view(), name='online-attribution-train')
]
