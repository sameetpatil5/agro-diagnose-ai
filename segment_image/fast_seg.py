import cv2
import numpy as np


def extract_leaf(image_path, output_path="out1_leaf.png"):
    """Extracts the leaf by automatically detecting and removing the background."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image!")
        return None

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's threshold to create a binary mask
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use GrabCut for fine-tuned segmentation
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = image * mask2[:, :, np.newaxis]

    # Save the extracted leaf
    cv2.imwrite(output_path, result)
    print(f"Processed image saved to {output_path}")
    return output_path


# Example usage:
image_path = r"C:\Users\Rehan\Desktop\CMR Dataset\113499.jpg"
extract_leaf(image_path)
