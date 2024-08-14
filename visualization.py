import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentation_functions import *  # Import all segmentation functions

def visualize_segmentation(sample_image_path):

# Select a sample image
    sample_image_path = os.path.join("\11th semester\MAH\Image Processing\Chosen Dataset\train\Afghan", os.listdir("\11th semester\MAH\Image Processing\Chosen Dataset\train\Afghan")[0])
    sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
    sample_image = cv2.resize(sample_image, (128, 128))

    # List of segmentation techniques and their names
    segmentation_functions = [
        (None, "Original"),
        (apply_clipping_thresholding, "Clipping & Thresholding"),
        (apply_digital_negative, "Digital Negative"),
        (apply_contrast_stretching, "Contrast Stretching"),
        (apply_canny_edge_detection, "Canny Edge Detection"),
        (apply_kmeans_segmentation, "K-means Clustering"),
        (apply_edge_based_segmentation, "Edge-based Segmentation"),
        (apply_image_sharpening, "Sharpening"),
        (apply_otsu_segmentation, "Otsu's Segmentation")
    ]

    # Plotting
    plt.figure(figsize=(15, 15))
    for i, (func, name) in enumerate(segmentation_functions, 1):
        if func:
            result_image = func(sample_image)
        else:
            result_image = sample_image

        plt.subplot(4, 3, i)
        plt.imshow(result_image, cmap='gray')
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
