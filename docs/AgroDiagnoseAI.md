# Research Document: Training and Testing CNN Models for Plant Disease Detection

## Introduction
This document outlines the steps required to train and test Convolutional Neural Network (CNN) models for plant disease detection. It provides a detailed plan for data preprocessing, model training, and evaluation, ensuring reproducibility and consistency across experiments. Additionally, references and datasets are provided for further exploration.

---

## Training Notebook Outline

### 1. Dataset Preparation
#### Datasets
- **PlantVillage Dataset (2016)**
  - Original: [GitHub Repository](https://github.com/spMohanty/PlantVillage-Dataset)
  - Mirror: [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data?status=pending)
- **PlantDoc Dataset for Benchmarking (2020)**
  - [GitHub Repository](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)

#### Steps:
1. Download datasets and organize into `train`, `validation`, and `test` folders.
2. Visualize class distribution and dataset splits.

#### Visualizations:
- Dataset details (e.g., number of images per class).
- Sample images for each class.

---

### 2. Image Preprocessing
#### Steps:
1. **Data Augmentation:**
   - Brightness, contrast, and sharpness adjustments.
   - Rotations (90°, 180°, 270°), vertical/horizontal flips.
   - Noise addition (Gaussian), PCA jittering.
2. **Image Enhancement:**
   - Retain original colors.
   - Convert to grayscale.
   - Segment images to remove backgrounds.

#### Visualizations:
- Augmented images with descriptions of augmentations applied.

---

### 3. Model Training
#### Custom CNN Models:
1. **Custom CNN 1:**
   - Total Parameters: 7,842,762
   - Architecture details.
2. **Custom CNN 2:**
   - Total Parameters: {our model}
   - Architecture details.

#### Transfer Learning Models:
1. **MobileNet**
2. **InceptionV3**
3. **EfficientNetB4/B5**
4. **ResNet50**

#### Steps:
1. Load pre-trained weights (for transfer learning).
2. Fine-tune models on the dataset.
3. Use appropriate loss functions (e.g., categorical cross-entropy).
4. Train with a learning rate scheduler and early stopping.

#### Visualizations:
- Model architecture.
- Training and validation accuracy/loss graphs.
- Confusion matrices.

---

### 4. Final Model Comparison
#### Steps:
1. Compare all models based on accuracy, precision, recall, and F1 score.
2. Identify the best-performing model.

#### Visualizations:
- Combined accuracy and loss graphs.
- Confusion matrix comparisons.

---

## Testing Notebook Outline

### 1. Dataset Preparation for Testing
#### Steps:
1. Organize test images into appropriate folders.
2. Preprocess images to match training dimensions (e.g., 128x128).

#### Visualizations:
- Sample test images.

---

### 2. Model Evaluation
#### Steps:
1. Load trained models.
2. Predict on test images.
3. Generate confusion matrices and classification reports.

#### Visualizations:
- Confusion matrix.
- Precision, recall, and F1 score for each class.

---

### 3. Single Image Prediction
#### Steps:
1. Load a single image.
2. Preprocess the image (resize, normalize).
3. Predict the class and display results.

#### Visualizations:
- Input image with predicted class label.

---

## References
1. **PlantVillage Dataset:** [GitHub Repository](https://github.com/spMohanty/PlantVillage-Dataset)
2. **PlantVillage Dataset Mirror:** [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data?status=pending)
3. **PlantDoc Dataset:** [GitHub Repository](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)
4. **Pre-Trained Model Guide:** [Analytics Vidhya Blog](https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/)
5. **InceptionV3 Example:** [Kaggle Notebook](https://www.kaggle.com/code/kmkarakaya/transfer-learning-for-image-classification)
6. **MobileNet Example:** [Kaggle Notebook](https://www.kaggle.com/code/lucasar/vs-vs-convnet-inception-xception-mobilenet)
7. **Deep Learning Plant Diseases:** [GitHub Repository](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases)
8. **Preprocessing Notes and Visualizations:** [Frontiers in Plant Science](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2020.01082/full)

