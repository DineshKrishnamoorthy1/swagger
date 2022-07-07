from django.urls import path

from agadia.views import Agadia

urlpatterns = [
    path('', Agadia.as_view(), name='agadia-attribution'),
]
