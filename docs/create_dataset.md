# Notes for Plant Disease Detection and Classification

## Dataset Structure

### 1. Plant Identification (Model 1: LeafIdentifier)
Organizes images based on plant type:
```
Dataset/
├── PlantIdentification/
│   ├── tomato/
│   │   ├── image1.jpg                # healthy
│   │   ├── image2.jpg                # diseased
│   │   ...
│   ├── potato/
│   │   ├── image1.jpg                # healthy
│   │   ├── image2.jpg                # diseased
│   │   ...
```

### 2. Disease Classification (Model 2: DiseaseClassifier)
Organizes images based on plant type and disease status:
```
Dataset/
├── DiseaseClassification/
│   ├── tomato/
│   │   ├── healthy/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ...
│   │   ├── early_blight/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ...
│   │   ├── late_blight/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ...
│   ├── potato/
│   │   ├── healthy/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ...
│   │   ├── black_leg/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ...
│   │   ├── leaf_roll/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ...
```

### 3. Direct Classification (Model 3: UnifiedClassifier)
Combines plant type and disease class into a single label:
```
Dataset/
├── DirectClassification/
│   ├── tomato_early_blight/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ...
│   ├── tomato_late_blight/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ...
│   ├── potato_black_leg/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ...
│   ├── potato_leaf_roll/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ...
```

### Dataset Sources
- Original dataset: [Kaggle Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- PlantVillage dataset:
  - Data: [GitHub Repository](https://github.com/salathegroup/plantvillage_deeplearning_paper_dataset)
  - Code: [GitHub Repository](https://github.com/salathegroup/plantvillage_deeplearning_paper_analysis)
  - Additional images: [PlantVillage Website](https://www.plantvillage.org/en/plant_images)
- PlantDoc dataset for benchmarking: [GitHub Repository](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)

### Notes on Dataset Preparation
- Use **leaves only** for disease detection.
- Create a **custom dataset** if necessary.
- Augment images:
  - Brightness, contrast, sharpness adjustments.
  - Rotations (90°, 180°, 270°), symmetry (vertical/horizontal).
  - Add noise (e.g., Gaussian), PCA jittering.
- Enhance images:
  - Keep original colors.
  - Create grayscale versions.
  - Segment images (remove backgrounds).

## Model Training

### Transfer Learning
- Use pre-trained models like GoogLeNet or AlexNet.
- Fine-tune hyperparameters:
  - Solver type: **Stochastic Gradient Descent**.
  - Base learning rate: **0.005**.
  - Learning rate policy: **Step** (decrease by a factor of 10 every 30/3 epochs).
  - Momentum: **0.9**.
  - Weight decay: **0.0005**.
  - Gamma: **0.1**.
  - Batch size: **24** (GoogLeNet), **100** (AlexNet).

### Model Evaluation
- Metrics:
  - Accuracy, Precision, Recall, F1 Score.
  - Confusion Matrix.
- Visualizations:
  - Convergence graphs (metrics vs. epochs).
  - Comparison of models with and without data augmentation.
  - Comparison of optimization algorithms.
  - Class Activation Maps (CAMs): Highlight areas influencing classification decisions.

### Benchmarking
- Compare against PlantDoc dataset results.
- Use other metrics like recognition performance and model convergence rates.

## Visualization and Reporting
- Showcases:
  - Original image of each class.
  - Augmented images with labels describing augmentations.
  - Dataset details (train/test/validation splits, class balance).
  - Model architecture and GPU configuration.
- Tables and Graphs:
  - Accuracy comparison with other models.
  - Confusion matrix (tabular and graphical).
  - Performance with/without dense connections and augmentation.
  - Activation visualizations (e.g., CAMs).

## Key References
- Augmentation techniques: [Frontiers in Plant Science](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2020.01082/full)
- PlantVillage datasets:
  - [GitHub Repository 1](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)
  - [GitHub Repository 2](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases)
- New Plant Disease Dataset: [Embrapa Dataset](https://www.redape.dados.embrapa.br/dataset.xhtml?persistentId=doi:10.48432/XA1OVL)

---

### Links for Backup

1. **Original Dataset**: [Kaggle Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. **PlantVillage Dataset**:
   - Data: [GitHub Repository](https://github.com/salathegroup/plantvillage_deeplearning_paper_dataset)
   - Code: [GitHub Repository](https://github.com/salathegroup/plantvillage_deeplearning_paper_analysis)
   - Additional images: [PlantVillage Website](https://www.plantvillage.org/en/plant_images)
3. **PlantDoc Dataset for Benchmarking**: [GitHub Repository](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)
4. **PlantVillage Data**:
   - [Mendeley Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)
   - [GitHub Repository (Raw Color)](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)
   - [GitHub Repository](https://github.com/spMohanty/PlantVillage-Dataset/tree/master)
5. **Additional Resources**:
   - [Embrapa Dataset](https://www.redape.dados.embrapa.br/dataset.xhtml?persistentId=doi:10.48432/XA1OVL) has pests involved
   - [Frontiers in Plant Science Article](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2020.01082/full)
   - [GitHub Repository for Deep Learning Plant Diseases](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases)

