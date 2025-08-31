from django.urls import path
from . import views

app_name = 'chat'

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.UserLoginView.as_view(), name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('change-password/', views.change_password, name='change_password'),
    path('ai-chat/', views.ai_chat, name='ai_chat'),
    path('api/chat/', views.process_chat_message, name='process_chat_message'),
    
    # Dataset Management URLs
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/upload/', views.dataset_upload, name='dataset_upload'),
    path('datasets/<int:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('datasets/<int:dataset_id>/edit/', views.dataset_edit, name='dataset_edit'),
    path('datasets/<int:dataset_id>/delete/', views.dataset_delete, name='dataset_delete'),
    path('datasets/<int:dataset_id>/download/', views.dataset_download, name='dataset_download'),
    
    # Test endpoint for debugging AJAX issues
    path('api/test-ajax/', views.test_ajax_endpoint, name='test_ajax_endpoint'),
    
    # Test endpoint for database creation
    path('api/test-database/', views.test_database_creation, name='test_database_creation'),
] 