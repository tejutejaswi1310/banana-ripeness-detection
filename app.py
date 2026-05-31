import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import local_binary_pattern

model = joblib.load("banana_model.pkl")
encoder = joblib.load("label_encoder.pkl")

IMG_SIZE = 128
radius = 3
n_points = 8 * radius

def extract_color_features(image):
    hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])

    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    hist = hist / np.sum(hist)
    return hist

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

st.title("🍌 Banana Ripeness Detection")
st.write("Upload a banana image to predict whether it is Unripe, Ripe, or Overripe.")

uploaded_file = st.file_uploader(
    "Choose a banana image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    color_feat = extract_color_features(img)
    texture_feat = extract_texture_features(img)

    combined_feat = np.concatenate([color_feat, texture_feat])
    prediction = model.predict([combined_feat])

    result = encoder.inverse_transform(prediction)[0]

    st.success(f"Prediction: {result.upper()}")