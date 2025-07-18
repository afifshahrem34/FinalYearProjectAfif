from django.urls import path

from . import views

urlpatterns = [

    path('', views.home, name="home"),
    path('login/', views.login, name="login"),
    path('signup/', views.signup, name="signup"),
    path('catalog/', views.catalog, name="catalog"),
    path('live-search/', views.sneaker_search, name='live_search'),
    path('cart/', views.cart, name="cart"),
    path('checkout/', views.checkout, name="checkout"),
    path('update_item/', views.update_item, name="update_item"),
    path('sneaker/<int:sneaker_id>/', views.sneaker_detail, name='sneaker_detail'),
    path('outfit/', views.outfit, name="outfit"),
    path('contact/', views.contact_us, name='contact'),
    path('logout/', views.logout_view, name='logout'),
    path('verify-otp/<int:customer_id>/', views.verify_otp, name='verify_otp'),
    path('forgot-password/', views.forgot_password, name='forgot_password'),
    path('reset-password/<str:token>/', views.reset_password_view, name='reset_password'),
    path('my-orders/', views.my_orders, name='my_orders')

]
