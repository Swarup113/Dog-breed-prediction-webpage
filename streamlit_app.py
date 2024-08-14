# client/streamlit_app.py

import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("Digital Image Processing")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    technique = st.selectbox("Choose any technique", [
        "Original",
        "Clipping & Thresholding",
        "Digital Negative",
        "Contrast Stretching",
        "Canny Edge Detection",
        "K-means Clustering",
        "Edge-based Segmentation",
        "Sharpening",
        "Otsu's Segmentation",

    ])

    if st.button("Apply"):
        files = {'image': uploaded_image.getvalue()}
        response = requests.post("http://localhost:5000/segment", files=files, data={'technique': technique})

        segmented_image = Image.open(BytesIO(response.content))

        st.image(segmented_image, caption="Segmented Image", use_column_width=True)
