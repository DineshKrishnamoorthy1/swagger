"""halo_py_service URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include
from django.urls import path
from drf_yasg.utils import swagger_auto_schema
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

from paper_itemization.PaperItemSchema import PaperSchema

schema_view = get_schema_view(
   openapi.Info(
      title="API",
      default_version='v1',
      description="Test description",
      terms_of_service="https://www.google.com/policies/terms/",
      contact=openapi.Contact(email="contact@snippets.local"),
      license=openapi.License(name="BSD License"),
   ),
   public=True,
   permission_classes=[permissions.AllowAny],
)



context_path = 'copro/'

urlpatterns = [
    path(context_path + 'admin/', admin.site.urls),
    path(context_path + 'paper-itemization', include('paper_itemization.urls')),
    path(context_path + 'asset-attribution', include('asset_attribution_model.urls')),
    path(context_path + 'handwritten-attribution', include('handwritten.urls')),
    # path(context_path + 'online-attribution', include('online_attribution.urls')),
    path(context_path + 'online-attribution-train', include('attribution_train.urls')),
    path(context_path + 'agadia-attribution', include('agadia.urls')),
    path('api/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),

]
