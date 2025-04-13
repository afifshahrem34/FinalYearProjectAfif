from django.urls import path

from . import views

urlpatterns = [
        #Leave as empty string for base url
	path('', views.home, name="home"),
	path('login/', views.login, name="login"),
	path('signup/', views.signup, name="signup"),
	path('catalog/', views.catalog, name="catalog"),
	path('cart/', views.cart, name="cart"),

]