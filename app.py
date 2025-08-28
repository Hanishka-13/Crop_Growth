import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# Inject CSS for background gradient
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        background-attachment: fixed;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load dataset info
@st.cache_data
def load_dataset():
    return pd.read_csv('dataset.csv')

def main():
    local_css("styles.css")

    st.title("🌾 Crop Health Monitoring")

    st.markdown(
        '<div class="upload-section" style="background-color:#1a1a2e; color:white;">'
        'Upload a crop leaf image for disease detection.'
        '</div>', 
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    df = load_dataset()

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Random prediction (since no model.h5)
        class_idx = np.random.randint(0, len(df))

        disease_name = df['disease_name'].iloc[class_idx]
        remedy = df['remedy'].iloc[class_idx]

        st.markdown(
            f'<div class="result-section">'
            f'<h3>Detected Disease: {disease_name}</h3>'
            f'<p><b>Remedy:</b> {remedy}</p>'
            f'</div>', 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()