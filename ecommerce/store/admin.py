from django.contrib import admin

from .models import *

admin.site.register(Customer)
admin.site.register(Sneaker)
admin.site.register(Size)
admin.site.register(SneakerSize)
admin.site.register(Order)
admin.site.register(OrderItem)
admin.site.register(ShippingAddress)