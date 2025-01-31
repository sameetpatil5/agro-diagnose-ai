import os
import cv2
from process_leaf import preprocess_leaf


def segment_dataset(dataset_dir, output_dir):
    """Preprocesses all leaf images in a dataset and organizes them into class-based folders."""

    if not os.path.exists(dataset_dir):
        print("Dataset directory does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_path):
            continue  # Skip non-folder items

        os.makedirs(output_class_path, exist_ok=True)

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            if not os.path.isfile(image_path):
                continue

            output_image_path = os.path.join(output_class_path, image_name)
            preprocess_leaf(image_path, output_image_path)

    print(f"Segmentation complete. Processed images are saved in {output_dir}")


# Example usage:
if __name__ == "__main__":
    dataset_dir = "segment_image/test_dir/"
    output_dir = "segment_image/test_out_dir/"
    segment_dataset(dataset_dir, output_dir)
