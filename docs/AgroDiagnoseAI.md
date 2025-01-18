# Research Document: Training and Testing CNN Models for Plant Disease Detection

## Introduction
This document outlines the steps required to train and test Convolutional Neural Network (CNN) models for plant disease detection. It provides a detailed plan for data preprocessing, model training, and evaluation, ensuring reproducibility and consistency across experiments. Additionally, references and datasets are provided for further exploration.

---

## Training Notebook Outline

### 1. Dataset Preparation

Using a custom dataset for multi model multi class classification is not a valid idea.
The current model without any alterations already achieves an accuracy of **97%**, which will be hard to beat while changing the entire structure of the dataset.

Plant Village Dataset @2016 is the most widely used dataset for plant disease classification using leaves.  It has over 50k images, which none of the datasets offer.

Datasets for Multi-model classification are non-existent. Hence the multi-model approach is scrapped for time-being.

A new dataset PlantDoc Dataset @2020 is available with relatively decent size and could be used for benchmarking.

#### Datasets
- **PlantVillage Dataset (2016)**
  - Original: [GitHub Repository](https://github.com/spMohanty/PlantVillage-Dataset)
  - Mirror: [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data?status=pending)
- **PlantDoc Dataset for Benchmarking (2020)**
  - [GitHub Repository](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)

#### Steps:
1. Download datasets and organize into `train`, `validation`, and `test` folders.
2. Visualize class distribution and dataset splits.

Dividing the dataset into 3 groups is important so as to maintain high accuracy 
Almost all the papers and articles I referred to have a description about the dataset to check for class distribution and to get an idea of how to preprocessing them for better accuracy.

#### Visualizations:
- Dataset details (e.g., number of images per class).
- Sample images for each class.

---

### 2. Image Preprocessing

The model referred form the tutorial does not perform any preprocessing on the images. Essentially with preprocessing I am referring to augmenting images to create variety and avoid overfitting.

General techniques include altering Brightness, contrast, sharpness, rotations, noise addition, jittering. Further enhancements may also include gray scaling and segmenting images but they need to be tested for individual model and compared to reach an conclusion weather or not to use them in the final training model.  

#### Steps:
1. **Data Augmentation:**
   - Brightness, contrast, and sharpness adjustments.
   - Rotations (90°, 180°, 270°), vertical/horizontal flips.
   - Noise addition (Gaussian), PCA jittering.
2. **Image Enhancement:**
   - Retain original colors.
   - Convert to grayscale.
   - Segment images to remove backgrounds.

It is important to visualize the augmentation performed on the images we can only push the model as far as a human can go. after preprocessing if a human couldn't classify the disease then its unlikely that the model will be able to.

Although we don't have right resources to check HLP(Human Level Performance) and compare them but its an important point to consider when training.

#### Visualizations:
- Augmented images with descriptions of augmentations applied.

---

### 3. Model Training

Even Training a simple CNN model is a much more time consuming process ~0.33 GPU hours. Which is significant because its possible we will need to training multiple modals. This is important because the current model has only 7M parameters but while using transfer learning the parameters scale significantly around 24M parameters which is 3x the normal CNN.

Right now the Plan is you take inspirations form the models online form research papers and the layers used in the transfer learning models to craft a custom CNN with relatively low parameters. But we do need to test the transfer learning models especially MobileNet, InceptionV3, EfficientNetB4. ResNet50 has been proved to be inefficient consistently in many articles and papers. But we can still test it if we get time.

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

It is important we handle the final layers of the model properly ass they largely determine the overall accuracy. Determining the loss function and properly mapping the classes are needed to be taken care of.

#### Steps:
1. Load pre-trained weights (for transfer learning).
2. Fine-tune models on the dataset.
3. Use appropriate loss functions (e.g., categorical cross-entropy).
4. Train with a learning rate scheduler and early stopping.

Certain visualization are mandatory architecture, Training and validation accuracy/loss graph and confusion matrix will help us finalize the model. Just looking at the numerical accuracy wont be great idea. Ideally we need the validation curve to be close to the accuracy curve or higher.

#### Visualizations:
- Model architecture.
- Training and validation accuracy/loss graphs.
- Confusion matrices.

---

### 4. Final Model Comparison

Once we have multiple models we will compare together. Calculate basic metrics accuracy, precision, recall, and F1 score. 

#### Steps:
1. Compare all models based on accuracy, precision, recall, and F1 score.
2. Identify the best-performing model.

Plot a combined graph of accuracy and loss of all the models to get an idea and finalize the production model.

#### Visualizations:
- Combined accuracy and loss graphs.
- Confusion matrix comparisons.

---

## Testing Notebook Outline

We will perform testing individually on each model and combine the results later for comparision.

### 1. Dataset Preparation for Testing

The preprocessing steps performed here will be required to perform on all the images which will be used for inference.

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

