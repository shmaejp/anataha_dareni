from django.urls import path
from . import views

urlpatterns = [
    path('', views.image_upload, name='imageupload'),
    # path('login/', views.Login.as_view(), name='login'),
    # path('logout/', views.Logout.as_view(), name='logout'),
    # path('signup/', views.signup, name='signup'),
]