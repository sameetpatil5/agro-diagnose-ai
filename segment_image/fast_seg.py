from fastapi import FastAPI, File, UploadFile, Response
import cv2
import numpy as np
import io

app = FastAPI()


def preprocess_leaf(image_bytes: bytes):
    """Processes an image by removing the background using OpenCV, without saving to disk."""
    
    # Convert bytes to NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None

    # Convert to grayscale and apply Otsu's threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use GrabCut for fine segmentation
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = image * mask2[:, :, np.newaxis]

    # Encode processed image back to bytes
    _, buffer = cv2.imencode(".png", result)
    return io.BytesIO(buffer)  # Return as BytesIO object


@app.post("/preprocess-image/")
async def process_image(file: UploadFile = File(...)):
    """Handles image upload, processes it in memory, and returns the modified image."""
    image_bytes = await file.read()  # Read uploaded file bytes
    processed_image_io = preprocess_leaf(image_bytes)

    if processed_image_io is None:
        return {"error": "Image processing failed"}

    # Return the processed image as a response
    return Response(content=processed_image_io.getvalue(), media_type="image/png")
