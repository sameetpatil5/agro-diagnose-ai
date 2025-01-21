import os
import random
import shutil

# Define paths
original_dataset_path = "datasets/Plant_Disease_Dataset"
new_dataset_path = "datasets/cropped_plant_village_dataset"

# Define the number of images per class
images_per_class = 50


def crop_folder(source_folder, destination_folder, num_images):
    """
    Crop a dataset folder to include only a fixed number of images per class.
    """
    for class_folder in os.listdir(source_folder):
        class_path = os.path.join(source_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Create destination class folder
        dest_class_path = os.path.join(destination_folder, class_folder)
        os.makedirs(dest_class_path, exist_ok=True)

        # Get all images in the class folder
        images = [
            f
            for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ]

        # Shuffle and select the desired number of images
        selected_images = random.sample(images, min(len(images), num_images))

        # Copy selected images to the destination folder
        for img in selected_images:
            src_img_path = os.path.join(class_path, img)
            dest_img_path = os.path.join(dest_class_path, img)
            shutil.copy(src_img_path, dest_img_path)


def copy_folder(source_folder, destination_folder):
    """
    Copy an entire folder without modifications.
    """
    shutil.copytree(source_folder, destination_folder)


# Process train folder
print("Processing train folder...")
train_source = os.path.join(original_dataset_path, "train")
train_dest = os.path.join(new_dataset_path, "train")
os.makedirs(train_dest, exist_ok=True)
crop_folder(train_source, train_dest, images_per_class)

# Process valid folder
print("Processing valid folder...")
valid_source = os.path.join(original_dataset_path, "valid")
valid_dest = os.path.join(new_dataset_path, "valid")
os.makedirs(valid_dest, exist_ok=True)
crop_folder(valid_source, valid_dest, images_per_class)

# Copy test folder without modifications
print("Copying test folder...")
test_source = os.path.join(original_dataset_path, "test")
test_dest = os.path.join(new_dataset_path, "test")
copy_folder(test_source, test_dest)

print("Dataset processing completed!")
print(f"Cropped dataset saved at: {new_dataset_path}")
