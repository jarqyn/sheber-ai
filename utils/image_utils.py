from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from utils.color_utils import rgb_to_hex

def load_image(image_path):
    img = Image.open(image_path)
    return img

def save_image(image, path):
    image.save(path)

def resize_image(img, target_width):
    original_size = img.size
    aspect_ratio = original_size[1] / original_size[0]
    target_height = int(target_width * aspect_ratio)
    img = img.resize((target_width, target_height))
    return img

def blur_image(img, blur_radius):
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def draw_contours_and_labels(numbered_image, edges):
    contour_image = Image.new('RGB', numbered_image.shape[:2][::-1], (255, 255, 255))
    draw = ImageDraw.Draw(contour_image)

    for i in range(numbered_image.shape[0]):
        for j in range(numbered_image.shape[1]):
            if edges[i, j]:
                draw.point((j, i), fill=(0, 0, 0))

    return contour_image

