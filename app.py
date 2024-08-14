# server/app.py

from flask import Flask, request, jsonify
import cv2
import numpy as np
from segmentation_functions import *  # Import all segmentation functions

app = Flask(__name__)


@app.route('/segment', methods=['POST'])
def segment_image():
    # Get the image and technique from the request
    image_file = request.files['image']
    technique = request.form.get('technique')

    # Convert the image to OpenCV format
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Apply the chosen segmentation technique
    if technique == "Clipping & Thresholding":
        segmented_image = apply_clipping_thresholding(image)
    elif technique == "Digital Negative":
        segmented_image = apply_digital_negative(image)
    elif technique == "Contrast Stretching":
        segmented_image = apply_contrast_stretching(image)
    elif technique == "Canny Edge Detection":
        segmented_image = apply_canny_edge_detection(image)
    elif technique == "K-means Segmentation":
        segmented_image = apply_kmeans_segmentation(image)
    elif technique == "Edge-based Segmentation":
        segmented_image = apply_edge_based_segmentation(image)
    elif technique == "Sharpening":
        segmented_image = apply_image_sharpening(image)
    elif technique == "Otsu's Segmentation":
        segmented_image = apply_otsu_segmentation(image)

    else:
        # If no known technique is selected, return the original image
        segmented_image = image

    # Convert the segmented image back to PNG format for sending as a response
    _, buffer = cv2.imencode('.png', segmented_image)

    # Return the segmented image as a response
    return buffer.tobytes(), 200, {'Content-Type': 'image/png'}


if __name__ == "__main__":
    app.run(port=5000)
