import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your best trained model
MODEL_PATH = r'models/inceptionv3_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload a brain MRI to classify the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = img_array / 255.0 # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)
    with col2:
        st.subheader("Prediction:")
        st.success(f"The model predicts a **{predicted_class}** tumor.")
        st.subheader("Confidence Score:")
        st.info(f"**{confidence:.2f}%**")