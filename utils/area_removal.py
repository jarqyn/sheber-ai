import numpy as np
from scipy import ndimage

def remove_small_areas(numbered_image, min_size=100):
    cleaned_image = np.copy(numbered_image)
    unique_labels = np.unique(numbered_image)
    for label in unique_labels:
        mask = (numbered_image == label)
        labeled, num_features = ndimage.label(mask)
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        too_small = sizes < min_size
        for small_label in range(1, num_features + 1):
            if too_small[small_label - 1]:
                cleaned_image[labeled == small_label] = 0
    return cleaned_image

