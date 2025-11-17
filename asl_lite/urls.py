# asl_site\urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from translator.views import index
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('translator.urls')),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.STATIC_URL,
        document_root=r"D:\PycharmProjects\asl_lite\translator\static"
    )
