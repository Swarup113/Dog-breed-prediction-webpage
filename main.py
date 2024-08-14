import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
from segmentation_functions import *
from visualization import *

# Function to load and preprocess images
def load_data(directory):
    labels = []
    data = []
    breed_classes = ["Afghan", "African Wild Dog", "Airedale", "Basenji", "Basset", "Beagle", "Bermaise"]

    for breed in breed_classes:
        breed_dir = os.path.join(directory, breed)

        for image_name in os.listdir(breed_dir):
            image_path = os.path.join(breed_dir, image_name)
            image = Image.open(image_path)

            # Resize and normalize the image
            image = image.resize((128, 128))
            image_arr = np.array(image) / 255.0

            # Flatten the image array and append to data
            data.append(image_arr.flatten())
            labels.append(breed)

    return np.array(data), np.array(labels)

# Load training, testing, and validation data
train_data, train_labels = load_data("\11th semester\MAH\Image Processing\Chosen Dataset\train")
test_data, test_labels = load_data("\11th semester\MAH\Image Processing\Chosen Dataset\test")
valid_data, valid_labels = load_data("\11th semester\MAH\Image Processing\Chosen Dataset\valid")

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(train_data, train_labels)
rf_preds = rf.predict(test_data)
print("Random Forest Classifier Report:")
print(classification_report(test_labels, rf_preds))

# Support Vector Classifier
svc = SVC()
svc.fit(train_data, train_labels)
svc_preds = svc.predict(test_data)
print("\nSupport Vector Classifier Report:")
print(classification_report(test_labels, svc_preds))

#Retrain the Model
def load_data_with_segmentation(directory, segmentation_function=None):
    labels = []
    data = []
    breed_classes = ["Afghan", "African Wild Dog", "Airedale", "Basenji", "Basset", "Beagle", "Bermaise"]

    for breed in breed_classes:
        breed_dir = os.path.join(directory, breed)

        for image_name in os.listdir(breed_dir):
            image_path = os.path.join(breed_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))

            # Apply the segmentation technique if provided
            if segmentation_function:
                image = segmentation_function(image)

            # Flatten the image array and append to data
            data.append(image.flatten())
            labels.append(breed)

    return np.array(data), np.array(labels)

# Training Evaluation

segmentation_techniques = [
    (None, "Original"),
    (apply_clipping_thresholding, "Clipping & Thresholding"),
    (apply_digital_negative, "Digital Negative"),
    (apply_contrast_stretching, "Contrast Stretching"),
    (apply_canny_edge_detection, "Canny Edge Detection")
]

segmentation_techniques += [
    (apply_kmeans_segmentation, "K-means Clustering"),
    (apply_edge_based_segmentation, "Edge-based Segmentation"),
    (apply_image_sharpening, "Sharpening"),
    (apply_otsu_segmentation, "Otsu's Segmentation")
]

for func, name in segmentation_techniques:
    print(f"Processing with {name} Technique...")

    # Load training and testing data with the segmentation technique
    train_data, train_labels = load_data_with_segmentation("\11th semester\MAH\Image Processing\Chosen Dataset\train", func)
    test_data, test_labels = load_data_with_segmentation("\11th semester\MAH\Image Processing\Chosen Dataset\test", func)

    # Random Forest Classifier
    rf = RandomForestClassifier()
    rf.fit(train_data, train_labels)
    rf_preds = rf.predict(test_data)
    print(f"\nRandom Forest Classifier Report ({name}):")
    print(classification_report(test_labels, rf_preds))

    # Support Vector Classifier
    svc = SVC()
    svc.fit(train_data, train_labels)
    svc_preds = svc.predict(test_data)
    print(f"\nSupport Vector Classifier Report ({name}):")
    print(classification_report(test_labels, svc_preds))
    print("-" * 50)



for func, name in segmentation_techniques:
    print(f"Processing with {name} Technique...")

    # Load training and testing data with the Image processing technique
    train_data, train_labels = load_data_with_segmentation("\11th semester\MAH\Image Processing\Chosen Dataset\train", func)
    test_data, test_labels = load_data_with_segmentation("\11th semester\MAH\Image Processing\Chosen Dataset\test", func)

    # Random Forest Classifier
    rf = RandomForestClassifier()
    rf.fit(train_data, train_labels)
    rf_preds = rf.predict(test_data)
    print(f"\nRandom Forest Classifier Report ({name}):")
    print(classification_report(test_labels, rf_preds))

    # Support Vector Classifier
    svc = SVC()
    svc.fit(train_data, train_labels)
    svc_preds = svc.predict(test_data)
    print(f"\nSupport Vector Classifier Report ({name}):")
    print(classification_report(test_labels, svc_preds))
    print("-" * 50)

