import numpy as np
from sklearn.cluster import KMeans

def cluster_image(img_array, n_colors):
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)
    clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]
    return clustered_pixels.reshape(img_array.shape).astype(np.uint8), kmeans

