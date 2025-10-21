from django.urls import path

from facts import views


app_name = 'facts'

urlpatterns = [
    path('', views.HomeView.as_view(), name="home"),
    path('analyze/', views.AnalyzeView.as_view(), name="analyze")
    ]