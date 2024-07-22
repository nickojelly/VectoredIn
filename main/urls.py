"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'main'
urlpatterns = [
    path('', views.plot_view, name='homepage'),
    path('plot/', views.plot_view, name='plot_view'),
    path('update_plot_data/<str:x_text>/<str:y_text>/<str:z_text>/<int:k>/<int:n>/', views.update_plot_data, name='update_plot_data'),
    path('update_plot_data/<str:x_text>/<str:y_text>/', views.update_plot_data, name='update_plot_data'),
    path('get_plot_data/<str:x_text>/<str:y_text>/<str:z_text>/', views.get_plot_data, name='get_plot_data'),
    path('get_point_summary/', views.get_point_summary, name='get_point_summary'),
    path('get_point_calculations/', views.get_point_calculations, name='get_point_calculations'),
    path('generate_plot_summary/', views.generate_plot_summary, name='generate_plot_summary'),
    path('get_alignment_summary/', views.get_alignment_summary, name='get_alignment_summary'),
    path('get_related_roles/', views.get_related_roles, name='get_related_roles'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
