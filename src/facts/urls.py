from django.urls import path

from facts import views


app_name = 'facts'

urlpatterns = [
    path('', views.HomeView.as_view(), name="home"),
    path('analyze/', views.TextAnalyze.as_view(), name="analyze")
    ]