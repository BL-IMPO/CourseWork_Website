from django.urls import path

from facts import views


app_name = 'facts'

urlpatterns = [
    path('', views.HomeView.as_view(), name="home"),
    path('analyze/', views.AnalyzeView.as_view(), name="analyze"),
    path('features/', views.FeaturesView.as_view(), name="features"),
    path('support/', views.SupportView.as_view(), name="support"),
    path('documentation/', views.DocumentationView.as_view(), name="documentation"),
    path('about/', views.AboutView.as_view(), name="about"),
    path('contact/', views.ContactView.as_view(), name="contact"),
    path('privacy/', views.PrivacyView.as_view(), name="privacy"),
    ]