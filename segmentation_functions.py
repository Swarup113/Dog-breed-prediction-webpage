import cv2
import numpy as np
from skimage import exposure

# 1. Clipping and Thresholding
def apply_clipping_thresholding(image):
    _, thresholded = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    return thresholded

# 2. Digital Negative
def apply_digital_negative(image):
    return 255 - image

# 3. Contrast Stretching
def apply_contrast_stretching(image):
    p2, p98 = np.percentile(image, (2, 98))
    return exposure.rescale_intensity(image, in_range=(p2, p98))

# 4. Canny Edge Detection
def apply_canny_edge_detection(image):
    return cv2.Canny(image, 100, 200)

# 5. Clustering based Image Segmentation (K-means)
def apply_kmeans_segmentation(image):
    image_flat = image.reshape((-1, 1))
    image_flat = np.float32(image_flat)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2  # number of clusters
    _, labels, centers = cv2.kmeans(image_flat, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    return np.uint8(segmented_image)

# 6. Edge-based Image Segmentation (Sobel Operator)
def apply_edge_based_segmentation(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return np.hypot(sobelx, sobely)

# 7. Sharpening
def apply_image_sharpening(image, alpha=1.5):
    # Create the Laplacian sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float32)

    # Apply the Laplacian sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, cv2.CV_64F, kernel)

    # Add the sharpened image back to the original image after scaling by alpha
    sharpened_image = image + alpha * sharpened_image

    # Ensure pixel values are in the valid range [0, 255]
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

    return sharpened_image

# 8. Otsu’s Image Segmentation
def apply_otsu_segmentation(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

