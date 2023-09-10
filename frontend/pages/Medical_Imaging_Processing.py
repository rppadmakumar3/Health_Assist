# frontend.py

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from transformers import pipeline

pipe = pipeline("image-classification", model="nickmuchi/vit-finetuned-chest-xray-pneumonia")

# Define the FastAPI backend URL
backend_url = "http://medical-imaging:8004"

st.title("Medical Imaging Processing")

data_folder = st.text_input('Dataset Folder', value='./medical-imaging/data')

# Model training button (existing functionality)
if st.button("Train Model"):
    st.text("Training in progress...")

    # Send a POST request to the FastAPI backend for model training (existing code)

# Image prediction section
st.header("Medical Image Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Add a button for users to submit their image
submit_button = st.button("Submit")

st.write("Please upload an image of a chest X-ray and click 'Submit' to get predictions.")

# Use the user input to make predictions when the button is clicked
if submit_button:
    if uploaded_file:
        # Load the image from the file uploader
        image = Image.open(uploaded_file)

        # Use the pipeline to make predictions directly
        result = pipe(image)

        # Display all the results in a table view
        if result:
            st.write("Predictions:")
            st.table(result)
    else:
        st.warning("Please upload an image of a chest X-ray to get predictions.")