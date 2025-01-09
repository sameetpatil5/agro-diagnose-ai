import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


# Load the model once for efficiency
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/trained_plant_disease_model.keras")


model = load_model()


# Tensorflow Model Prediction
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# Sidebar
st.sidebar.title("🌱 Navigation")
app_mode = st.sidebar.radio(
    "Select Page", ["🏠 Home", "📚 About", "🔍 Disease Recognition"]
)

# Main Page
if app_mode == "🏠 Home":
    st.title("🌿 Plant Disease Recognition System")
    st.image("Plant_demo/home_page.jpeg", use_container_width=True)
    st.markdown(
        """
        ### Welcome to Plant Health Companion 🌱
        Effortlessly diagnose plant diseases and take timely actions to save your crops.
        #### Features:
        - 🌟 AI-powered disease detection
        - 📊 Insightful analysis and treatment suggestions
        - 🌍 Accessible to farmers everywhere
        Navigate to **Disease Recognition** to upload a plant image and get started!
        """
    )

elif app_mode == "📚 About":
    st.title("📚 About the Project")
    st.markdown(
        """
        This project uses a dataset derived from the **PlantVillage** dataset.
        ### Dataset Details:
        - **Training Images**: 70,295
        - **Validation Images**: 17,572
        - **Test Images**: 33
        - **Classes**: 38 plant diseases, including healthy leaves.
        ### Technology Stack:
        - **TensorFlow** for deep learning
        - **Streamlit** for the user interface
        - **Google Cloud** for secure image storage
        """
    )

elif app_mode == "🔍 Disease Recognition":
    st.title("🔍 Disease Recognition")
    st.markdown("Upload an image of a plant leaf to identify any disease.")

    test_image = st.file_uploader("📂 Upload Image:", type=["jpg", "png", "jpeg"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict"):
            st.spinner("Analyzing...")
            result_index = model_prediction(test_image)

            # Class labels
            class_name = [
                "Apple___Apple_scab",
                "Apple___Black_rot",
                "Apple___Cedar_apple_rust",
                "Apple___healthy",
                "Blueberry___healthy",
                "Cherry_(including_sour)___Powdery_mildew",
                "Cherry_(including_sour)___healthy",
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                "Corn_(maize)___Common_rust_",
                "Corn_(maize)___Northern_Leaf_Blight",
                "Corn_(maize)___healthy",
                "Grape___Black_rot",
                "Grape___Esca_(Black_Measles)",
                "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                "Grape___healthy",
                "Orange___Haunglongbing_(Citrus_greening)",
                "Peach___Bacterial_spot",
                "Peach___healthy",
                "Pepper,_bell___Bacterial_spot",
                "Pepper,_bell___healthy",
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Potato___healthy",
                "Raspberry___healthy",
                "Soybean___healthy",
                "Squash___Powdery_mildew",
                "Strawberry___Leaf_scorch",
                "Strawberry___healthy",
                "Tomato___Bacterial_spot",
                "Tomato___Early_blight",
                "Tomato___Late_blight",
                "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot",
                "Tomato___Spider_mites Two-spotted_spider_mite",
                "Tomato___Target_Spot",
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                "Tomato___Tomato_mosaic_virus",
                "Tomato___healthy",
            ]
            st.success(f"🌟 Detected: **{class_name[result_index]}**")
            st.balloons()
