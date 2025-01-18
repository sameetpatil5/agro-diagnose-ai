### Dataset Information

1. **Original Dataset Used in the Tutorial**  
   [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

2. **PlantVillage Dataset**  
   - Data and Code:
     - [Dataset](https://github.com/salathegroup/plantvillage_deep_learning_paper_dataset)
     - [Code](https://github.com/salathegroup/plantvillage_deep_learning_paper_analysis)
   - Additional Image Data: [PlantVillage Image Data](https://www.plantvillage.org/en/plant_images)
   - Alternative Sources:
     - [Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)
     - [Raw Color Dataset](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)
     - [PlantVillage Dataset Repository](https://github.com/spMohanty/PlantVillage-Dataset/tree/master)

3. **Benchmarking Dataset**  
   - **PlantDoc Dataset**: Used for benchmarking classification models.  
     - Repository: [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset?tab=readme-ov-file)  
     - Research Paper: "PlantDoc: A Dataset for Visual Plant Disease Detection" (CoDS-COMAD 2020)

4. **Other Datasets**  
   - **Plant Classification Datasets** (Not Useful):  
     - [Dataset 1](https://huggingface.co/datasets/Taquito07/plant_classification_v2/viewer)  
     - [Dataset 2](https://huggingface.co/datasets/Mirkat/Plant_Classification/viewer)
   - **Healthy vs. Unhealthy Dataset**: Too small for practical use.  
     - [Dataset](https://huggingface.co/datasets/ayerr/plant-disease-classification)
   - **PlantVillage with Background**: Useful for accuracy improvement.  
     - [Repository](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases?tab=readme-ov-file)
   - **New Plant Disease Dataset**: Not widely used.  
     - [Dataset](https://www.redape.dados.embrapa.br/dataset.xhtml?persistentId=doi:10.48432/XA1OVL)

---

### Methodology

1. **Approach**  
   - Focus on **leaves only** for disease detection.
   - Use **transfer learning** with pre-trained models (e.g., GoogLeNet, AlexNet).  
     - Ensure proper configuration and tuning of hyperparameters.

2. **Image Configuration**  
   - Enhance image quality through techniques like:  
     - Original color, grayscale, and segmentation (removing background).  

3. **Image Augmentation**  
   - Examples of augmentations:  
     - Brightness (high/low), contrast (high/low), sharpness (high/low).  
     - Rotations (90°, 180°, 270°), symmetry (vertical/horizontal).  
     - Add Gaussian noise, PCA jittering.  
   - Visualization reference:  
     - [Figure 2: Augmentation of a grape leaf disease image](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2020.01082/full)

4. **Hyperparameter Tuning**  
   - Solver: Stochastic Gradient Descent (SGD).  
   - Learning Rate: 0.005, with step policy (reduce by factor of 10 every 10 epochs).  
   - Momentum: 0.9, Weight Decay: 0.0005, Gamma: 0.1.  
   - Batch Size:  
     - GoogLeNet: 24  
     - AlexNet: 100  

5. **Performance Metrics**  
   - Benchmark against other models.  
   - Compare accuracy and other metrics in a table.  
   - Analyze convergence: Plot metrics (e.g., accuracy) vs. epochs.  
   - Use a **confusion matrix** for evaluation.  
     - Plot confusion matrix for your model.  
     - Compare performance with and without data augmentation.  

6. **Activation Visualization**  
   - Generate **Class Activation Maps (CAM)** to understand model decisions.  
     - Examples:  
       - Anthracnose, Brown Spot, Mites, Black Rot, Downy Mildew, Leaf Blight, Healthy Leaves.  

---

### Final Outputs

1. **Dataset Details**  
   - Train/Test/Validation split.  
   - Number of images per class.  
   - Balance of classes.  

2. **Model Structure**  
   - Detailed architecture (layer-by-layer information).  
   - GPU configuration used for training.  

3. **Visualization Requirements**  
   - Heatmaps of class activation.  
   - Augmented image samples with labels.  

4. **Comparison and Analysis**  
   - Recognition performance across models.  
   - Optimization algorithms comparison (graph plot).  
   - Your model's performance with and without dense connections.  
