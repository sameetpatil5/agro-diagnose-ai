### Training the Best CNN Model

#### Dataset Selection
- **Valid Datasets for Testing:**
  - **Plant Village Dataset (2016):**
    - Original link: [GitHub](https://github.com/spMohanty/PlantVillage-Dataset)
    - Mirror link: [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data?status=pending)
  - **PlantDoc Dataset for Benchmarking (2020):**
    - [GitHub Link](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)

#### Visualizations
- **Dataset Details:**
  - Train/Test/Validation splits.
  - Class balance.
- **Original Images:**
  - Display representative images for each class.

---

### Image Preprocessing

#### Dataset Splits
- Train/Validation/Test splits.

#### Preprocessing Steps for Leaves Only
- **Image Augmentation:**
  - Adjust brightness, contrast, and sharpness.
  - Apply rotations (90°, 180°, 270°) and symmetry (vertical/horizontal).
  - Add noise (e.g., Gaussian), PCA jittering.
- **Image Enhancement:**
  - Retain original colors.
  - Generate grayscale versions.
  - Segment images (remove backgrounds).

#### Visualizations
- Augmented images with labels describing the augmentations.

---

### Training

#### Custom Models
- **Custom CNN 1:**
  - Total parameters: 7,842,762 (29.92 MB)
  - Trainable parameters: 7,842,762 (29.92 MB)
  - Non-trainable parameters: 0 (0.00 B)
- **Custom CNN 2:**
  - Details to be added.

#### Transfer Learning
- **Top Pre-Trained Models:**
  - **ResNet50**
  - **InceptionV3:** [Kaggle Notebook](https://www.kaggle.com/code/kmkarakaya/transfer-learning-for-image-classification)
  - **EfficientNet (B4 or B5)**
  - **MobileNet:** [Kaggle Notebook](https://www.kaggle.com/code/lucasar/vs-vs-convnet-inception-xception-mobilenet#Detection-of-Cats,-Dogs-and-Wild-Animals-using-Convolutional-Neural-Networks)

---

### Visualizations (Individual)
- **Model Architecture:**
  - Display architecture and GPU configuration.
- **Performance Metrics:**
  - Confusion matrix.
  - Train/Validation accuracy and loss graphs.
  - Accuracy without augmentation.

---

### Final Model Comparison

#### Combined Visualizations
- Comparison of all models based on accuracy, parameters, and performance.

---

### References and Links
- **Comparative Study with Visualizations:** [GitHub Repository](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases)
- **Preprocessing Notes and Visualization Instructions:** [Frontiers Article](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2020.01082/full)

