import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Classes (must match training order)
CLASSES = ["trash", "plastic", "paper", "metal", "glass", "cardboard"]
IMG_SIZE = (160, 160)

# Model path (in repo root)
MODEL_PATH = "garbage_classifier_300.tflite"

# Load TFLite model once
@st.cache_resource
def load_interpreter():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Streamlit UI ---
st.title("♻️ Garbage Classifier (TFLite)")
st.write("Upload an image of waste (trash, plastic, paper, metal, glass, cardboard) and let the model predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_class = CLASSES[np.argmax(preds)]
    confidence = np.max(preds)

    st.markdown(f"### 🏷 Prediction: **{pred_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence:.2f}**")
