import os
import logging
import numpy as np
from config import *
from PIL import Image
from utils.image_utils import load_image, save_image, resize_image, blur_image, draw_contours_and_labels
from utils.color_utils import rgb_to_hex
from utils.area_removal import remove_small_areas
from clustering.kmeans_clustering import cluster_image

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.FileHandler("image_processing.log"),  
        logging.StreamHandler() 
    ]
)

def image_to_numbers_and_outline(image_path, n_colors=7, blur_radius=2, min_area_size=50):
    logging.info(f"Starting image processing for {image_path}")

    try:
        img = load_image(image_path)
        logging.info("Image loaded successfully")
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return

    img = resize_image(img, 2000)
    logging.info("Image resized")

    if blur_radius > 0:
        img = blur_image(img, blur_radius)
        logging.info(f"Blurred image with radius {blur_radius}")

    img_array = np.array(img)
    clustered_pixels, kmeans = cluster_image(img_array, n_colors)
    logging.info(f"Image clustered into {n_colors} colors")

    unique_colors = np.unique(clustered_pixels.reshape(-1, 3), axis=0)
    color_map = {tuple(color): i + 1 for i, color in enumerate(unique_colors)}
    logging.debug("Color map created")

    numbered_image = np.zeros(img_array.shape[:2], dtype=int)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            color = tuple(clustered_pixels[i, j])
            numbered_image[i, j] = color_map[color]

    numbered_image = remove_small_areas(numbered_image, min_size=min_area_size)
    logging.info(f"Small areas removed (min size: {min_area_size})")

    colored_image_array = np.zeros(img_array.shape, dtype=np.uint8)
    for i in range(numbered_image.shape[0]):
        for j in range(numbered_image.shape[1]):
            color = unique_colors[numbered_image[i, j] - 1]
            colored_image_array[i, j] = color

    filled_image = Image.fromarray(colored_image_array)
    save_image(filled_image, FILLED_IMAGE_PATH)
    logging.info(f"Filled image saved to {FILLED_IMAGE_PATH}")

    edges = np.zeros_like(numbered_image)
    for i in range(1, numbered_image.shape[0] - 1):
        for j in range(1, numbered_image.shape[1] - 1):
            if numbered_image[i, j] != numbered_image[i + 1, j] or numbered_image[i, j] != numbered_image[i, j + 1]:
                edges[i, j] = 1

    contour_image = draw_contours_and_labels(numbered_image, edges)
    save_image(contour_image, OUTPUT_IMAGE_PATH)
    logging.info(f"Contour image saved to {OUTPUT_IMAGE_PATH}")

    with open(COLOR_CODES_FILE, 'w') as f:
        f.write("Colors used in the image (HEX):\n")
        for i, color in enumerate(unique_colors):
            hex_color = rgb_to_hex(color.astype(int))
            f.write(f"Color {i + 1}: {hex_color}\n")
    logging.info(f"Color codes saved to {COLOR_CODES_FILE}")

if __name__ == "__main__":
    os.makedirs('images', exist_ok=True)  
    logging.info("Starting main process")
    image_to_numbers_and_outline(DEFAULT_IMAGE_PATH, n_colors=10, blur_radius=6, min_area_size=100)
    logging.info("Image processing completed")

