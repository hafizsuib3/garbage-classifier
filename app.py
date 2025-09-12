import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load model
MODEL_PATH = os.path.join("model", "garbage_classifier_300.h5")
model = load_model(MODEL_PATH)

# Classes (must match training order)
CLASSES = ["trash", "plastic", "paper", "metal", "glass", "cardboard"]
IMG_SIZE = (160, 160)

st.title("♻️ Garbage Classifier")
st.write("Upload an image of waste (trash, plastic, paper, metal, glass, cardboard) and let the model predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize(IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Prediction
    preds = model.predict(x)
    pred_class = CLASSES[np.argmax(preds)]
    confidence = np.max(preds)

    st.markdown(f"### 🏷 Prediction: **{pred_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence:.2f}**")
