from django.urls import path

from . import views


urlpatterns = [
    path("graph", views.graph),
    path("contentmgmt", views.contentmgmt),

]