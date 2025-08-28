import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd

# Inject CSS for background gradient
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); /* Dark to light blue gradient */
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)


# Load CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load model function (replace with your actual model)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load dataset info
@st.cache_data
def load_dataset():
    return pd.read_csv('dataset.csv')

def main():
    local_css("styles.css")

    st.title("🌾 Crop Health Monitoring")

    st.markdown('<div class="upload-section">Upload a crop leaf image for disease detection.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    model = load_model()
    df = load_dataset()

    if uploaded_file and model is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess image for model
        img = image.resize((224, 224))  # Change size as per your model input
        img_array = np.array(img) / 255.0

        # If image is grayscale convert to RGB
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)

        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        st.write("🔍 Raw prediction scores:", prediction)  # Debug info
        class_idx = np.argmax(prediction, axis=1)[0]

        # Safeguard index in case CSV and model mismatch
        if class_idx >= len(df):
            st.error("Prediction index out of range of dataset.")
            return

        disease_name = df['disease_name'].iloc[class_idx]
        remedy = df['remedy'].iloc[class_idx]

        st.markdown(f'<div class="result-section"><h3>Detected Disease: {disease_name}</h3><p><b>Remedy:</b> {remedy}</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()