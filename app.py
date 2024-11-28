import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Classifier",
    page_icon=":medical_thermometer:",
    layout="wide"
)

# Load the pre-trained model
@st.cache_resource
def load_pneumonia_model():
    model = load_model('saved_models/pneumonia_cnn_model.h5')
    return model

# Preprocess the image
def preprocess_image(uploaded_file):
    # Read the image
    img = image.load_img(uploaded_file, target_size=(64, 64), color_mode='grayscale')
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Normalize the image
    img_array = img_array / 255.0
    
    # Expand dimensions to create batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Prediction function
def predict_pneumonia(model, img_array):
    # Make prediction
    prediction = model.predict(img_array)
    
    # Convert to percentage
    probability = prediction[0][0] * 100
    
    # Classify
    if probability > 50:
        return f"Pneumonia Detected (Probability: {probability:.2f}%)", probability
    else:
        return f"Normal (Probability: {(100-probability):.2f}%)", 100-probability

# Main Streamlit app
def main():
    # Title
    st.title("Pneumonia Classification from Chest X-Ray")
    
    # Sidebar
    st.sidebar.header("About the App")
    st.sidebar.info(
        "This app uses a Convolutional Neural Network (CNN) "
        "to classify chest X-ray images as Normal or Pneumonia."
    )
    
    # Load model
    model = load_pneumonia_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a Chest X-Ray Image", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Chest X-Ray", width=300)
        
        # Preprocess the image
        img_array = preprocess_image(uploaded_file)
        
        # Prediction
        if st.button("Classify Image"):
            # Make prediction
            result, probability = predict_pneumonia(model, img_array)
            
            # Display results
            st.subheader("Prediction Results")
            st.write(result)
            
            # Progress bar
            st.progress(int(probability)/100)

# Run the app
if __name__ == "__main__":
    main()
