import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ecommerce.settings')
django.setup()

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import cv2
import colorsys
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from django.conf import settings
import random

# Directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ai_model')
DATASET_DIR = os.path.join(MODEL_DIR, 'OutfitDataset')

# Load models and datasets once
feature_extractor = load_model(os.path.join(MODEL_DIR, 'feature_extractor.h5'))
final_model = load_model(os.path.join(MODEL_DIR, 'final_model.h5'))
categories = np.load(os.path.join(MODEL_DIR, 'dataset_categories.npy'))
colors = np.load(os.path.join(MODEL_DIR, 'dataset_colors.npy'))
features = np.load(os.path.join(MODEL_DIR, 'dataset_features.npy'))
labels = np.load(os.path.join(MODEL_DIR, 'dataset_labels.npy'))
paths = np.load(os.path.join(MODEL_DIR, 'dataset_paths.npy'))

# Image preprocessing configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def get_image_data_generators(data_dir):
    """Configure image data generators for training and validation."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, val_generator

def get_average_color(image_array, exclude_color=(255, 255, 0), tolerance=10):
    """Calculates the average color of non-background pixels."""
    diff = np.abs(image_array - np.array(exclude_color))
    mask = np.all(diff < tolerance, axis=-1)
    non_background_pixels = image_array[~mask]
    return np.mean(non_background_pixels, axis=0).astype(int) if len(non_background_pixels) > 0 else (0, 0, 0)

def replace_background(img_path, new_img_path, color=(0.0, 1.0, 1.0)):
    """Replaces background with a solid color using edge detection and contours."""
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = color

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    try:
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        cv2.imwrite(new_img_path, img)
        return

    contour_info = sorted([(c, cv2.contourArea(c)) for c in contours], key=lambda x: x[1], reverse=True)
    max_contour = contour_info[0][0]

    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour, 255)
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3).astype('float32') / 255.0
    img = img.astype('float32') / 255.0

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')
    cv2.imwrite(new_img_path, masked)

def classify_rgb_to_color_name(r, g, b):
    """Classifies an RGB value into a color name using HSV thresholds."""
    hue, sat, val = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue *= 360

    if val <= 0.15: return 'black'
    if sat <= 0.08 and val > 0.85: return 'white'
    if sat <= 0.12 and 0.15 < val <= 0.85: return 'gray'
    if 10 < hue <= 50 and 0.15 < sat <= 0.9 and 0.15 < val <= 0.9:
        return 'brown' if val <= 0.55 else 'tan'
    if hue <= 18 or hue > 340: return 'red'
    if 200 <= hue <= 260 and sat > 0.2: return 'navy' if val <= 0.7 else 'blue'
    if 65 < hue <= 165: return 'green'
    if 45 < hue <= 65: return 'yellow'
    if 20 < hue <= 40 and sat > 0.4 and val > 0.4: return 'orange'
    if 295 < hue <= 340: return 'pink'
    if 265 < hue <= 295: return 'purple'
    if sat > 0.15:
        if 0 <= hue < 30: return 'reddish_brown'
        if 30 <= hue < 60: return 'yellowish_brown'
        if 60 <= hue < 90: return 'lime_green'
        if 90 <= hue < 120: return 'cyan_green'
        if 120 <= hue < 150: return 'aqua'
        if 150 <= hue < 180: return 'teal'
        if 180 <= hue < 210: return 'sky_blue'
        if 260 <= hue < 280: return 'magenta'
        if 280 <= hue < 300: return 'fuchsia'
    return 'unknown'

def get_dominant_colors_kmeans(image_rgb_pixels, n_clusters=5):
    """Uses K-Means clustering to find dominant colors in the image."""
    if not image_rgb_pixels:
        return []

    pixels = np.array(image_rgb_pixels).reshape(-1, 3)
    n_clusters = min(n_clusters, len(pixels))
    if n_clusters == 0:
        return []

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(pixels)
    centroids = kmeans.cluster_centers_.astype(int)
    counts = Counter(kmeans.labels_)

    ranked_colors = []
    for i in range(n_clusters):
        rgb = tuple(centroids[i])
        color_name = classify_rgb_to_color_name(*rgb)
        ranked_colors.append((color_name, counts[i], rgb))

    return sorted(ranked_colors, key=lambda x: x[1], reverse=True)

def classify_shoe_color(image_path, n_dominant_colors=3):
    """Classifies shoe color after background removal and KMeans analysis."""
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_img_path = os.path.join(temp_dir, 'temp_shoe_for_color.jpg')

    replace_background(image_path, temp_img_path)

    img = cv2.imread(temp_img_path)
    if img is None:
        return 'unknown'

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    exclude_color_rgb = (255, 255, 0)
    tolerance = 25

    filtered_pixels = [
        tuple(pixel) for pixel in img_rgb.reshape(-1, 3)
        if not all(abs(int(pixel[i]) - exclude_color_rgb[i]) < tolerance for i in range(3))
    ]

    if not filtered_pixels:
        return 'unknown'

    dominant_colors_info = get_dominant_colors_kmeans(filtered_pixels, n_clusters=n_dominant_colors)
    significant_colors = [info for info in dominant_colors_info if info[0] not in ['yellow', 'unknown']]

    if not significant_colors:
        return dominant_colors_info[0][0] if dominant_colors_info else 'unknown'

    neutral_colors = ['black', 'white', 'gray', 'brown', 'tan', 'beige', 'navy']
    for color_name, _, _ in significant_colors:
        if color_name not in neutral_colors:
            return color_name

    return significant_colors[0][0]

def classify_outfit_color(rgb_tuple):
    """Classifies outfit item color based on HSV."""
    r, g, b = rgb_tuple
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h_deg = h * 360

    if v < 0.15: return 'black'
    if s < 0.1 and v > 0.85: return 'white'
    if s < 0.15 and 0.15 <= v <= 0.85: return 'gray'
    if (0 <= h_deg < 20 or h_deg > 340) and s > 0.2: return 'red'
    if 20 <= h_deg < 50 and s > 0.3: return 'orange' if v > 0.7 else 'brown'
    if 50 <= h_deg < 70 and s > 0.3: return 'yellow'
    if 70 <= h_deg < 160 and s > 0.2: return 'dark_green' if v < 0.4 else 'green'
    if 160 <= h_deg < 260 and s > 0.2: return 'navy' if v < 0.35 else ('light_blue' if v > 0.7 else 'blue')
    if 260 <= h_deg < 320 and s > 0.2: return 'dark_purple' if v < 0.35 else ('light_purple' if v > 0.7 else 'purple')
    if 320 <= h_deg <= 340 and s > 0.2: return 'pink'
    if 10 <= h_deg <= 40 and 0.2 <= s <= 0.7 and 0.2 <= v <= 0.7: return 'tan' if v > 0.6 else 'brown'
    if 10 <= h_deg <= 60 and 0.05 <= s <= 0.25 and 0.7 <= v <= 0.95: return 'beige'
    return 'other'

color_compatibility = {
    'black': ['white', 'gray', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'orange', 'brown', 'tan', 'beige'],
    'white': ['black', 'gray', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'orange', 'brown', 'tan', 'beige', 'navy'],
    'gray': ['black', 'white', 'blue', 'pink', 'purple', 'green', 'red', 'yellow', 'orange', 'brown', 'tan', 'beige', 'navy'],
    'navy': ['white', 'gray', 'red', 'blue', 'green', 'yellow', 'pink', 'orange', 'brown', 'tan', 'beige', 'black'],
    'brown': ['tan', 'beige', 'white', 'black', 'blue', 'green', 'red', 'yellow', 'orange', 'gray', 'navy'],
    'tan': ['brown', 'beige', 'white', 'black', 'blue', 'green', 'red', 'yellow', 'orange', 'gray', 'navy'],
    'beige': ['brown', 'tan', 'white', 'black', 'blue', 'green', 'red', 'yellow', 'orange', 'gray', 'navy'],
    'red': ['black', 'white', 'gray', 'navy', 'blue', 'green', 'yellow', 'brown', 'tan', 'beige'],
    'blue': ['white', 'black', 'gray', 'navy', 'brown', 'tan', 'beige', 'red', 'yellow', 'green', 'pink', 'purple', 'orange'],
    'green': ['white', 'black', 'gray', 'blue', 'brown', 'tan', 'beige', 'yellow', 'red', 'orange', 'navy'],
    'yellow': ['blue', 'black', 'white', 'gray', 'green', 'brown', 'navy', 'red', 'orange', 'purple'],
    'pink': ['blue', 'gray', 'white', 'black', 'purple', 'green', 'navy', 'beige'],
    'purple': ['white', 'black', 'gray', 'blue', 'pink', 'green', 'yellow', 'navy'],
    'orange': ['blue', 'white', 'black', 'gray', 'green', 'red', 'yellow', 'brown', 'navy'],
    'other': ['black', 'white', 'gray', 'navy', 'brown']
}

def recommend_outfit(query_image_path, top_n=3):
    """Generates outfit recommendations based on shoe image."""
    if feature_extractor is None or features is None:
        return {'error': 'Models or dataset features are not loaded.'}

    # Detect shoe color
    shoe_color = classify_shoe_color(query_image_path)
    compatible_colors = color_compatibility.get(shoe_color, ['black', 'white', 'gray', 'navy', 'brown', 'tan', 'beige'])

    # Extract shoe style features
    shoe_img = load_img(query_image_path, target_size=IMG_SIZE)
    shoe_array = img_to_array(shoe_img) / 255.0
    shoe_feature = feature_extractor.predict(np.expand_dims(shoe_array, axis=0), verbose=0)

    categories_list = ['Accessories', 'Tops', 'Bottoms']
    recommendations = {'shoe_color': shoe_color, 'categories': {}}

    for category in categories_list:
        # Filter items in current category
        indices = [i for i, label in enumerate(labels) if label.lower() == category.lower()]
        if not indices:
            recommendations['categories'][category] = []
            continue

        category_features = features[indices]
        category_paths = [paths[i] for i in indices]
        item_colors = [colors[i] for i in indices]

        # Style similarity (cosine)
        style_scores = cosine_similarity(shoe_feature, category_features)[0]

        # Color compatibility
        color_scores = []
        for color in item_colors:
            if color == shoe_color:
                color_scores.append(0.3)
            elif color in compatible_colors:
                color_scores.append(0.15)
            elif color in ['black', 'white', 'gray', 'brown', 'tan', 'beige', 'navy']:
                color_scores.append(0.05)
            else:
                color_scores.append(0)

        # Final score (style + color)
        final_scores = 0.85 * style_scores + 0.15 * np.array(color_scores)
        final_scores += np.random.uniform(0, 1e-5, size=final_scores.shape)  # Add noise to avoid ties

        top_k = np.argsort(final_scores)[::-1][:10]
        top_indices = sorted(random.sample(list(top_k), min(top_n, len(top_k))))

        recommendations['categories'][category] = [
            {
                'name': os.path.basename(category_paths[idx]),
                'path': category_paths[idx],
                'score': float(final_scores[idx] * 100),
                'color': item_colors[idx]
            } for idx in top_indices
        ]

    return recommendations

if __name__ == '__main__':
    test_image_path = os.path.join(BASE_DIR, 'static', 'images', 'sneaker_catalog_1.jpg')
    result = recommend_outfit(test_image_path)
    import pprint
    pprint.pprint(result)