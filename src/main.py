from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load ONNX model
MODEL_PATH = "../models/trained_plant_disease_model.onnx"
session = ort.InferenceSession(MODEL_PATH)

# Get model input shape dynamically
input_shape = session.get_inputs()[0].shape
image_size = (input_shape[2], input_shape[1])  # Ensure correct (width, height)

# Class Names
CLASS_NAMES = [
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

# Define the confidence threshold
CONFIDENCE_THRESHOLD = 0.6


# Function to preprocess image
def preprocess_image(image_bytes, image_size):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(image_size)
    image_array = np.array(image).astype(np.float32)

    # Normalize the image
    image_array /= 255.0  # Normalization to [0, 1]

    # Expand dimensions to match model input (batch size, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)  # (1, 128, 128, 3)

    return image_array


# FastAPI Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes, image_size)

        # Run inference
        inputs = {session.get_inputs()[0].name: image_array}
        outputs = session.run(None, inputs)

        # Get predicted class probabilities
        class_probabilities = outputs[0][0]

        # Get predicted class index and its probability
        result_index = np.argmax(class_probabilities)
        max_probability = class_probabilities[result_index]

        if max_probability >= CONFIDENCE_THRESHOLD:
            # Map index to class name
            predicted_class = CLASS_NAMES[result_index]
            return {
                "message": "Prediction successful.",
                "prediction": predicted_class,
                "confidence": float(max_probability),
            }
        else:
            return {
                "message": "Prediction confidence too low to determine.",
                "prediction": None,
                "confidence": float(max_probability),
            }

    except Exception as e:
        return {"error": str(e)}
