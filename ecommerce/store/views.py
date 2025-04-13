from django.shortcuts import render

def login(request):
	context = {}
	return render(request, 'store/login.html', context)

def signup(request):
	context = {}
	return render(request, 'store/signup.html', context)

def home(request):
	context = {}
	return render(request, 'store/home.html', context)

def catalog(request):
	context = {}
	return render(request, 'store/catalog.html', context)

def cart(request):
	context = {}
	return render(request, 'store/cart.html', context)