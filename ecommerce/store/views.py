from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from django.contrib.auth.password_validation import validate_password
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail
from datetime import timedelta
from ai_model.inference import recommend_outfit
from django.contrib.sites.shortcuts import get_current_site
from django.urls import reverse
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from django.conf import settings
from django.http import JsonResponse
from .models import *
from pathlib import Path
import os
import json
import random


def convert_to_static_path(original_path):
    # Extract just the filename
    filename = os.path.basename(original_path)
    category = 'Accessories' if 'Accessories' in original_path else \
        'Tops' if 'Tops' in original_path else \
            'Bottoms' if 'Bottoms' in original_path else 'Unknown'
    return f"/static/images/OutfitDataset/{category}/{filename}"


def verify_otp(request, customer_id):
    customer = get_object_or_404(Customer, id=customer_id)

    if request.method == 'POST':
        entered_otp = request.POST.get('otp')

        # Check if OTP matches and not expired
        if customer.otp_code == entered_otp:
            if timezone.now() <= customer.otp_expiry:
                customer.is_verified = True
                customer.otp_code = None
                customer.otp_expiry = None
                customer.save()
                messages.success(request, "Email verified successfully. You can now log in.")
                return redirect('login')
            else:
                messages.error(request, "OTP has expired. Please sign up again.")
                return redirect('signup')
        else:
            messages.error(request, "Invalid OTP. Please try again.")

    return render(request, 'store/verify_otp.html')


reset_tokens = {}


def forgot_password(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)

            if not user.customer.is_verified:
                messages.error(request, "This email is not verified.")
                return redirect('login')

            token = get_random_string(length=32)

            # Store token with expiry
            reset_tokens[token] = {
                'username': user.username,
                'expires_at': timezone.now() + timedelta(minutes=10)
            }

            reset_url = request.build_absolute_uri(f'/reset-password/{token}/')

            send_mail(
                'Reset Your Supa Snika Password',
                f"Hi {user.first_name}, click this link to reset your password:\n{reset_url}",
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False
            )

            messages.success(request, "Reset link has been sent to your email.")
            return redirect('login')

        except User.DoesNotExist:
            messages.error(request, "No account found with this email.")
            return redirect('login')


def reset_password_view(request, token):
    token_data = reset_tokens.get(token)

    # Invalid or expired token
    if not token_data or timezone.now() > token_data['expires_at']:
        messages.error(request, "Reset link expired.")
        reset_tokens.pop(token, None)  # Clean up
        return redirect('login')

    if request.method == 'POST':
        password = request.POST.get('password')
        confirm = request.POST.get('confirm_password')

        if password != confirm:
            messages.error(request, "Passwords do not match.")
            return redirect(request.path)

        try:
            username = token_data['username']
            user = User.objects.get(username=username)
            user.set_password(password)
            user.save()

            reset_tokens.pop(token, None)  # Cleanup used token
            messages.success(request, "Password reset successfully. You can now log in.")
            return redirect('login')
        except Exception as e:
            messages.error(request, "Something went wrong.")
            return redirect('login')

    return render(request, 'store/reset_password.html')


def signup(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        full_name = f"{first_name} {last_name}"
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        # Email format validation
        try:
            validate_email(email)
        except ValidationError:
            messages.error(request, "Please enter a valid email address.")
            return redirect('signup')

        # Password match check
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        # Check if user already exists
        if User.objects.filter(username=email).exists():
            messages.error(request, "Unable to create account. Try again or use a different email.")
            return redirect('signup')

        try:
            # Validate password strength
            validate_password(password)

            # Create user
            user = User.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )

            # Generate OTP
            otp = f"{random.randint(100000, 999999)}"
            otp_expiry = timezone.now() + timedelta(minutes=5)

            # Create customer profile
            customer = Customer.objects.create(
                user=user,
                name=full_name,
                email=email,
                otp_code=otp,
                otp_expiry=otp_expiry
            )

            # Build verification link
            current_site = get_current_site(request)
            verification_path = reverse('verify_otp', args=[customer.id])
            verification_url = f"http://{current_site.domain}{verification_path}"

            # Compose email
            email_subject = 'Your Supa Snika Verification Code'
            email_body = (
                f"Hi {full_name},\n\n"
                f"Your OTP is: {otp}\n"
                f"It expires in 5 minutes.\n\n"
                f"Click the link below to verify your email:\n"
                f"{verification_url}\n\n"
                f"Thank you for joining Supa Snika!\n"
                f"Having problems during registration? Contact us directly via:\n"
                f"Call: +60183178991 , Email: supasnikka@gmail.com"
            )

            send_mail(
                subject=email_subject,
                message=email_body,
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=[email],
                fail_silently=False,
            )

            messages.success(request, "Account created successfully. Please check your email to verify.")
            return render(request, 'store/signup.html', {
                'otp_redirect': True,
                'customer_id': customer.id,
                'products': Sneaker.objects.all(),
                'cartItems': 0,
            })


        except ValidationError as e:
            for error in e.messages:
                messages.error(request, error)
            return redirect('signup')
        except Exception as e:
            messages.error(request, "Something went wrong. Please try again.")
            return redirect('signup')

    # Cart handling
    cartItems = 0
    if request.user.is_authenticated:
        customer = request.user.customer
        order, _ = Order.objects.get_or_create(customer=customer, complete=False)
        cartItems = order.get_cart_items

    products = Sneaker.objects.all()
    context = {'products': products, 'cartItems': cartItems}
    return render(request, 'store/signup.html', context)


def login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)

        if user:
            if not user.customer.is_verified:
                messages.error(request, 'Please verify your email before logging in.')
                return redirect('login')
            auth_login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid email or password.')

    # Cart handling
    cartItems = 0
    if request.user.is_authenticated:
        customer = request.user.customer
        order, _ = Order.objects.get_or_create(customer=customer, complete=False)
        cartItems = order.get_cart_items

    products = Sneaker.objects.all()
    context = {'products': products, 'cartItems': cartItems}
    return render(request, 'store/login.html', context)


def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('login')


def home(request):
    if request.user.is_authenticated:
        customer = request.user.customer
        order, created = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
    else:
        # Create empty cart for now for non-logged in user
        items = []
        order = {'get_cart_total': 0, 'get_cart_items': 0}
        cartItems = order['get_cart_items']

    products = Sneaker.objects.all()
    context = {'products': products, 'cartItems': cartItems}
    return render(request, 'store/home.html', context)


def catalog(request):
    if request.user.is_authenticated:
        customer = request.user.customer
        order, created = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
    else:
        # Create empty cart for now for non-logged in user
        items = []
        order = {'get_cart_total': 0, 'get_cart_items': 0}
        cartItems = order['get_cart_items']

    products = Sneaker.objects.all()
    context = {'products': products, 'cartItems': cartItems}
    return render(request, 'store/catalog.html', context)


def sneaker_search(request):
    if request.method == 'GET':
        query = request.GET.get('term', '').strip()
        results = []
        if query:
            sneakers = Sneaker.objects.filter(sneaker_name__icontains=query)[:10]
            results = [{
                'id': s.id,
                'name': s.sneaker_name,
                'price': s.sneaker_price,
                'image': s.imageURL,
                'url': f"/sneaker/{s.id}/"
            } for s in sneakers]
        return JsonResponse({'data': results})


def update_item(request):
    data = json.loads(request.body)

    product_id = data.get('productId')
    action = data.get('action')
    size_id = data.get('sizeId')  # Get size from frontend

    print('Action:', action)
    print('Product ID:', product_id)
    print('Size ID:', size_id)

    customer = request.user.customer

    try:
        product = Sneaker.objects.get(id=product_id)
        selected_size = SneakerSize.objects.get(id=size_id)
    except Sneaker.DoesNotExist:
        return JsonResponse({'error': 'Sneaker not found'}, status=404)
    except SneakerSize.DoesNotExist:
        return JsonResponse({'error': 'Size not found'}, status=404)

    order, created = Order.objects.get_or_create(customer=customer, complete=False)

    order_item, created = OrderItem.objects.get_or_create(
        order=order,
        item=product,
        selected_size=selected_size
    )

    if action == 'add':
        if order_item.quantity < selected_size.stock:
            order_item.quantity += 1
            order_item.save()
            return JsonResponse({'status': 'success'})
        else:
            return JsonResponse(
                {'error': f'Only {selected_size.stock} left in stock.'}, status=400)
    elif action == 'remove':
        order_item.quantity -= 1
        if order_item.quantity <= 0:
            order_item.delete()
        else:
            order_item.save()

    return JsonResponse({'status': 'updated', 'message': 'Item removed or quantity updated'})


@login_required(login_url='login')
def cart(request):
    if request.user.is_authenticated:
        customer = request.user.customer
        order, created = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
    else:
        # Create empty cart for now for non-logged in user
        items = []
        order = {'get_cart_total': 0, 'get_cart_items': 0}
        cartItems = order['get_cart_items']

    context = {'items': items, 'order': order, 'cartItems': cartItems}
    return render(request, 'store/cart.html', context)


def checkout(request):
    errors = {}
    previous_shipping = None

    if request.user.is_authenticated:
        customer = request.user.customer
        order, created = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
        previous_shipping = ShippingAddress.objects.filter(customer=customer).order_by('-date_added').first()

        if request.method == 'POST':
            address = request.POST.get('address', '').strip()
            phone_number = request.POST.get('phone_number', '').strip()
            city = request.POST.get('city', '').strip()
            state = request.POST.get('state', '').strip()
            zipcode = request.POST.get('zipcode', '').strip()

            # Validation
            if not address:
                errors['address'] = 'Address is required.'
            if not phone_number:
                errors['phone_number'] = 'Phone Number is required.'
            if not city:
                errors['city'] = 'City is required.'
            if not state:
                errors['state'] = 'State is required.'
            if not zipcode:
                errors['zipcode'] = 'Zip Code is required.'

            if not errors:
                # Check stock availability
                for item in items:
                    sneaker_size = item.selected_size
                    if sneaker_size.stock < item.quantity:
                        messages.error(request, f"Not enough stock for {item.item.sneaker_name} (Size {sneaker_size.size.label}).")
                        return redirect('cart')

                # Save shipping
                ShippingAddress.objects.create(
                    customer=customer,
                    order=order,
                    address=address,
                    phone_number=phone_number,
                    city=city,
                    state=state,
                    zipcode=zipcode
                )

                # Complete order
                transaction_id = ''.join([str(random.randint(0, 9)) for _ in range(10)])
                order.transaction_id = transaction_id
                order.complete = True
                order.date_ordered = timezone.now()
                order.order_status = 'created'
                order.save()

                # Subtract stock
                for item in items:
                    sneaker_size = item.selected_size
                    sneaker_size.stock -= item.quantity
                    sneaker_size.save()

                # Email receipt - single message for all items
                subject = 'Supa Snika - Order Confirmation'
                message = (
                    f"Dear {request.user.first_name} {request.user.last_name},\n\n"
                    f"Thank you for your order!\n\n"
                    f"Order ID: {order.id}\n"
                    f"Date & Time: {order.date_ordered.astimezone().strftime('%d %B %Y, %I:%M %p')}\n"
                    f"Transaction ID: {transaction_id}\n\n"
                    f"Order Summary:\n\n"
                )

                for item in items:
                    message += f"- {item.item.sneaker_name} (Size: {item.selected_size.size.label}) x {item.quantity} = RM{item.get_total:.2f}\n"

                message += (
                    f"\nTotal Amount: RM{order.get_cart_total:.2f}\n\n"
                    f"Shipping to:\n\n{address}, {city}, {state}, {zipcode}\n\n"
                    f"Thanks again for shopping with Supa Snika!"
                )

                send_mail(
                    subject,
                    message,
                    settings.DEFAULT_FROM_EMAIL,
                    [request.user.email],
                    fail_silently=False
                )

                # Clear old incomplete carts
                Order.objects.filter(customer=customer, complete=False).exclude(id=order.id).delete()

                messages.success(request, 'Thank you! Your order has been placed successfully.')
                return redirect('my_orders')

    else:
        items = []
        order = {'get_cart_total': 0, 'get_cart_items': 0}
        cartItems = 0

    context = {
        'items': items,
        'order': order,
        'cartItems': cartItems,
        'errors': errors,
        'previous_shipping': previous_shipping
    }

    return render(request, 'store/checkout.html', context)


def sneaker_detail(request, sneaker_id):
    sneaker = get_object_or_404(Sneaker, id=sneaker_id)
    sneaker_sizes = sneaker.sneakersize_set.select_related('size').all()

    if request.user.is_authenticated:
        customer = request.user.customer
        order, _ = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
    else:
        items = []
        order = {'get_cart_total': 0, 'get_cart_items': 0}
        cartItems = order['get_cart_items']

    # AI Recommendation logic
    recommendation_result = None
    image_preview_url = None

    try:
        image_path = sneaker.sneaker_image.path
        image_preview_url = sneaker.sneaker_image.url
        recommendation_result = recommend_outfit(image_path)

        if recommendation_result and 'categories' in recommendation_result:
            for category in recommendation_result['categories']:
                for item in recommendation_result['categories'][category]:
                    item['static_path'] = convert_to_static_path(item['path'])

    except Exception as e:
        recommendation_result = {'error': str(e)}

    context = {
        'sneaker': sneaker,
        'sneaker_sizes': sneaker_sizes,
        'cartItems': cartItems,
        'recommendation': recommendation_result,
        'image_preview_url': image_preview_url,
    }

    return render(request, 'store/view.html', context)


def outfit(request):
    if request.user.is_authenticated:
        customer = request.user.customer
        order, _ = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
    else:
        items = []
        order = {'get_cart_total': 0, 'get_cart_items': 0}
        cartItems = order['get_cart_items']

    recommendation_result = None
    image_preview_url = None  # To show uploaded image in the template

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        # Save image to media/uploads/
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(image.name, image)
        image_path = os.path.join(upload_dir, filename)
        image_preview_url = os.path.join(settings.MEDIA_URL, 'uploads', filename).replace("\\", "/")

        try:
            recommendation_result = recommend_outfit(image_path)

            # Convert dataset paths to static or media URLs
            if recommendation_result and 'categories' in recommendation_result:
                for category in recommendation_result['categories']:
                    for item in recommendation_result['categories'][category]:
                        item['static_path'] = convert_to_static_path(item['path'])

        except Exception as e:
            recommendation_result = {'error': str(e)}

    products = Sneaker.objects.all()
    context = {
        'products': products,
        'cartItems': cartItems,
        'recommendation': recommendation_result,
        'image_preview_url': image_preview_url,
    }
    return render(request, 'store/ai_outfit.html', context)


def contact_us(request):
    # Preload cart info for context
    if request.user.is_authenticated:
        customer = request.user.customer
        order, _ = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
    else:
        items = []
        order = {'get_cart_total': 0, 'get_cart_items': 0}
        cartItems = order['get_cart_items']

    products = Sneaker.objects.all()

    # Handle form submission
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')  # user’s email
        subject = request.POST.get('subject', 'No Subject')
        message = request.POST.get('message')

        full_message = f"""
           A new inquiry from user:

           Name: {name}
           Email: {email}
           Subject: {subject}

           Message:
           {message}
           """

        try:
            # 1. Send to your support email
            send_mail(
                subject=f"[Supa Snika Inquiry] {subject}",
                message=full_message,
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=['supasnikka@gmail.com'],
                fail_silently=False,
            )

            # 2. Send a confirmation copy to the user
            send_mail(
                subject="We've received your message - Supa Snika",
                message=f"Hi {name},\n\nThank you for contacting Supa Snika. We’ve received your message:\n\n{message}\n\nWe'll reply to you as soon as possible!\n\n- Supa Snika Team",
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=[email],
                fail_silently=False,
            )

            messages.success(request, "Thank you for contacting us! We've sent a copy of your message to your email.")
        except Exception as e:
            print("EMAIL ERROR:", e)
            messages.error(request, "Email failed to send. Please try again.")

        return redirect('contact')

    context = {
        'products': products,
        'cartItems': cartItems
    }
    return render(request, 'store/contact_us.html', context)


def my_orders(request):
    customer = request.user.customer

    # Get all completed orders
    orders = Order.objects.filter(customer=customer, complete=True).order_by('-date_ordered')

    # For navbar/cart display
    current_order, _ = Order.objects.get_or_create(customer=customer, complete=False)
    cartItems = current_order.get_cart_items

    context = {
        'orders': orders,
        'cartItems': cartItems,
    }

    return render(request, 'store/my_orders.html', context)
