//Add to cart function
var updateBtns = document.getElementsByClassName('update-cart');

for (let i = 0; i < updateBtns.length; i++) {
    updateBtns[i].addEventListener('click', function () {
        const productId = this.dataset.product;
        const action = this.dataset.action;
        const sizeId = this.dataset.size;
        const stock = parseInt(this.dataset.stock) || 99; // fallback if missing
        const isCartUpdate = this.dataset.source === 'cart';  // <- detect if source is cart

        if (!sizeId) {
            let sizeId = this.dataset.size;
            sizeId = sizeSelect ? sizeSelect.value : '';
        }

        if (!sizeId) {
            alert('Please select a sneaker size before adding to cart.');
            return;
        }

        // Optional: prevent exceeding stock in cart
        if (action === 'add' && isCartUpdate) {
            const quantityElement = this.closest('.quantity')?.querySelector('.quantity');
            const currentQty = parseInt(quantityElement?.textContent?.trim() || '0');
            if (currentQty >= stock) {
                alert(`Only ${stock} left in stock`);
                return;
            }
        }

        updateUserOrder(productId, action, sizeId, isCartUpdate);
    });
}

function updateUserOrder(productId, action, sizeId, isCartUpdate) {
    console.log('Sending cart data...');

    fetch('/update_item/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken,
        },
        body: JSON.stringify({
            productId,
            action,
            sizeId
        })
    })
    .then((response) => {
        if (!response.ok) {
            return response.json().then(data => {
                alert(data.error || "Something went wrong.");
                throw new Error(data.error);
            });
        }
        return response.json();
    })
    .then((data) => {
        if (action === 'add') {
            if (!isCartUpdate) {
                // Update cart count manually without reloading
                const cartTotal = document.getElementById('cart-total');
                if (cartTotal) {
                    const currentCount = parseInt(cartTotal.textContent) || 0;
                    cartTotal.textContent = currentCount + 1;

                    // Animate the bump
                    cartTotal.classList.add('cart-bump');
                    setTimeout(() => cartTotal.classList.remove('cart-bump'), 300);
                }

                Swal.fire({
                    icon: 'success',
                    title: 'Added to cart',
                    showConfirmButton: false,
                    timer: 1000
                });
            } else {
                location.reload();
            }
        } else {
            location.reload(); // for 'remove'
        }
    })
    .catch((error) => {
        console.error('Cart update error:', error);
    });
}