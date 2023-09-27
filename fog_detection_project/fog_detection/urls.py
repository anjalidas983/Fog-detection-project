from django.urls import path 
from . import views
app_name='fog_detection'



urlpatterns = [

    path('',views.home,name='home'),
]