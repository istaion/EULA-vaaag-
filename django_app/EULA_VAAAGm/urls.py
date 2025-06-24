"""
URL configuration for EULA_VAAAGm project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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
from django.urls import path
from ask_ur_16th_mommy.views import HomeView, GenerateAudioView

from django.conf import settings  # ✅ nécessaire
from django.conf.urls.static import static  # ✅ nécessaire


app_name = 'eula_vaaagr'

urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('generate-audio/', GenerateAudioView.as_view(), name='generate_audio'),
    path('admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

