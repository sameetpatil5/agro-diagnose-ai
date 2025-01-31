import os
import cv2
import multiprocessing
from process_leaf import preprocess_leaf


def process_image(args):
    """Function to process a single image using multiprocessing."""
    image_path, output_image_path = args
    preprocess_leaf(image_path, output_image_path)


def segment_dataset(dataset_dir, output_dir):
    """Preprocesses all leaf images in a dataset using multiprocessing and organizes them into class-based folders."""

    if not os.path.exists(dataset_dir):
        print("Dataset directory does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    tasks = []  # List to store image processing tasks

    for part in os.listdir(dataset_dir):
        part_path = os.path.join(dataset_dir, part)
        if os.path.isdir(part_path):
            output_part_path = os.path.join(output_dir, part)

            for class_name in os.listdir(part_path):
                class_path = os.path.join(part_path, class_name)

                if not os.path.isdir(class_path):
                    continue  # Skip non-folder items

                output_class_path = os.path.join(output_part_path, class_name)
                os.makedirs(output_class_path, exist_ok=True)

                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)

                    if not os.path.isfile(image_path):
                        continue

                    output_image_path = os.path.join(output_class_path, image_name)
                    tasks.append((image_path, output_image_path))

    # Use multiprocessing Pool to parallelize image processing
    num_workers = min(
        multiprocessing.cpu_count(), len(tasks)
    )  # Use available CPU cores
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_image, tasks)

    print(f"Segmentation complete. Processed images are saved in {output_dir}")


# Example usage:
if __name__ == "__main__":
    dataset_dir = "datasets/cropped_plant_village_dataset"
    output_dir = "datasets/segmented_cropped_plant_village_dataset"
    segment_dataset(dataset_dir, output_dir)
