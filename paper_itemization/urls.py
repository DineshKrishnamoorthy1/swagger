from django.urls import path

from .views import PaperItemization

urlpatterns = [
    path('', PaperItemization.as_view(), name='paper_itemization'),
]
