#  Satellite Imagery Classification: Agricultural vs Non-Agricultural

##  Project Overview
This project focuses on **Satellite Land-Use Classification** using **Deep Learning**.  
A **Convolutional Neural Network (CNN)** is implemented to automatically classify satellite images into:

-  **Agricultural land**
-  **Non-Agricultural land**

The workflow includes automated data ingestion, advanced preprocessing with **real-time data augmentation**, and an optimized training strategy using **dynamic learning rate scheduling**.

---

##  Dataset & Visualizations

The dataset is organized using a **directory-based image pipeline**, where images are:
- Normalized
- Augmented (rotation, flipping, zooming)
- Batched efficiently for training

###  Sample Dataset
Representative samples from both classes:

<p align="center">
  <img src="samples/data_set.png" width="80%" alt="Dataset Samples"/>
</p>

###  Model Performance
Training quality and reliability are evaluated using:

- Accuracy & Loss curves
- Confusion Matrix

<p align="center">
  <img src="samples/training_accuracy_loss.png" width="45%" alt="Training Curves"/>
  <img src="samples/confusion_matrix.png" width="45%" alt="Confusion Matrix"/>
</p>

---

##  Key Technical Features

###  Asynchronous Data Ingestion
- Uses **httpx** for non-blocking dataset downloads
- Improves preprocessing speed and scalability

###  CNN Architecture
- Conv2D layers for spatial feature extraction  
- BatchNormalization for training stability  
- MaxPooling & GlobalAveragePooling2D for dimensionality reduction  

###  Smart Training Callbacks
- **EarlyStopping** → prevents overfitting  
- **ReduceLROnPlateau** → adaptive learning rate optimization  

###  Data Augmentation
Real-time transformations to enhance generalization:
- Horizontal & vertical flips
- Rotations
- Zoom operations

---

##  Installation & Usage

###  Clone the Repository

git clone https://github.com/ibrahim-mosaad/Satellite-Land-Classification.git
cd Satellite-Land-Classification


Install Dependencies

pip install -r requirements.txt

## Author

Ibrahim Mosaad

GitHub: @ibrahim-mosaad

AI Engineer | Computer Vision Enthusiast




