from django.urls import path
from explaining.views import *

app_name = "explaining"

urlpatterns = [
    path(r'', main_page, name="main_page"),
    # path(r'explanations', explanations, name='explanations')
]