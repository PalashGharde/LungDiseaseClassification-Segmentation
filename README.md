# Lung Disease Classification and Localization using Deep Learning  

## 📌 Overview  
This project implements **deep learning models** to classify and localize lung diseases using **chest X-ray images**. It utilizes **ConvNeXt** for multi-label disease classification and **Swin Transformer** for disease localization, improving diagnostic accuracy in medical imaging.  

---

## 🔍 Key Features  
- ✅ **Multi-label Classification**: ConvNeXt-based model achieving **80.11% AUC** on ChestX-ray14 dataset  
- ✅ **Disease Localization**: Swin Transformer-based localization achieving **79% IoU**  
- ✅ **Grad-CAM Visualization**: Helps interpret model predictions by highlighting important regions  

---

## 📂 Datasets Used  
- **[ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)** – 112,120 chest X-rays for classification  
- **[MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)** – 377,110 chest X-ray images for multi-task learning  
- **[VinDR-CXR](https://vindr.ai/datasets/vindr-cxr)** – 18,000 expert-annotated X-ray images for localization  

---

## 🏗 Model Architectures  

### 🏥 **1️⃣ Classification: ConvNeXt**  
![ConvNeXt Architecture](Images/ConvNext.png)
- A modern **CNN-based model** with residual connections and **layer normalization**  
- Trained on **ChestX-ray14** dataset to classify **14 lung diseases**  
- Achieved **80.11% AUC** on the test set  

### 🏥 **2️⃣ Localization: Swin Transformer**  
![Swin Transformer Architecture](Images/Swin_Transformer.png)
- A **hierarchical vision transformer** that captures spatial relationships  
- Trained on **VinDR-CXR** dataset for disease localization  
- Achieved **79% Intersection over Union (IoU)** in identifying diseased regions  

---

## 🚀 Results  

| Task           | Model            | Performance  |
|---------------|-----------------|-------------|
| Classification | ConvNeXt         | 80.11% AUC  |
| Localization  | Swin Transformer | 79% IoU     |

![Classification Result 1](Images/classification%20Results.png)
![Classification Result 2](Images/Classification%20results%202.png)

---

## 🔧 Implementation Steps  

### 1️⃣ **Data Collection and Preprocessing**  
- Download datasets: **ChestX-ray14, MIMIC-CXR, and VinDR-CXR**  
- Resize images (**224x224** for ConvNeXt, **1024x1024** for Swin Transformer)  
- Normalize pixel values between **0 and 1** for model training  
- Convert labels into **multi-hot encoding** (classification) and **bounding boxes** (localization)  
- Split into **training, validation, and test sets**  

---

### 2️⃣ **Model Selection and Architecture**  
#### 🔹 **Classification: ConvNeXt**  
- Uses **CNN-based residual blocks** for **multi-label classification**  
- Key modifications:  
  ✅ **Global Average Pooling Layer** – Reduces dimensionality  
  ✅ **Fully Connected Layer (ReLU activation)** – Enhances feature learning  
  ✅ **Dropout Layer** – Prevents overfitting  
  ✅ **Sigmoid Activation** – Outputs probability for each disease  

#### 🔹 **Localization: Swin Transformer**  
- **Hierarchical transformer-based model** for spatial feature extraction  
- Implements **shifted window attention mechanism** for better context awareness  
- Trained within a **Mask R-CNN framework** for **bounding-box localization**  

---

### 3️⃣ **Model Training and Hyperparameter Tuning**  
- **Optimizer**: **AdamW** for efficient weight updates  
- **Loss functions**:  
  🔹 **BCEWithLogitsLoss** – Classification  
  🔹 **IoU Loss** – Localization  
- **Training Hyperparameters**:  
  🔹 **Batch size** = 32  
  🔹 **Learning rate** = 0.0001  
  🔹 **Epochs** = 20 (Classification), **50+** (Localization)  
- **Regularization Techniques**:  
  🔹 **Dropout layers**  
  🔹 **Data augmentation** (rotation, scaling, flipping)  

---

### 4️⃣ **Model Evaluation and Performance Metrics**  
- **Classification (ConvNeXt)**: Evaluated using **AUC (Area Under the Curve)** – **80.11% AUC**  
- **Localization (Swin Transformer)**: Evaluated using **Intersection over Union (IoU)** – **79% IoU**  
---

## 📊 Performance Metrics  
- 📌 **AUC (Area Under Curve)** – Classification  
- 📌 **IoU (Intersection over Union)** – Localization  
- 📌 **Grad-CAM** – Class Activation Map for interpretability  

---

## 📝 Future Improvements  
🔹 Experiment with **multi-task learning** to improve classification and localization jointly  
🔹 Apply **self-supervised learning (DINO)** for better feature extraction  
🔹 Enhance **attention mechanisms** for more precise localization  

---

## 📜 References  
📌 **ConvNeXt**: [Paper](https://arxiv.org/abs/2201.03545)  
📌 **Swin Transformer**: [Paper](https://arxiv.org/abs/2103.14030)  

