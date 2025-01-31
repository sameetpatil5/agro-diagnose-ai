import cv2
import numpy as np
import os


def preprocess_leaf(image_path, output_path="preprocessed_leaf.png"):
    """Preprocesses a leaf image by removing the background using Otsu's thresholding and GrabCut."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's threshold to create a binary mask
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use GrabCut for better segmentation
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Create final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = image * mask2[:, :, np.newaxis]

    # Save preprocessed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"Preprocessed image saved to {output_path}")
    return output_path


# Example usage:
if __name__ == "__main__":
    # image_path = "sample_image_test.jpg"
    image_path = "/home/sam5io/sam_engineerings/AgroDiagnoseAI/segment_image/sample_image_test.jpg"

    preprocess_leaf(
        image_path,
        "/home/sam5io/sam_engineerings/AgroDiagnoseAI/segment_image/preprocessed_leaf.png",
    )
