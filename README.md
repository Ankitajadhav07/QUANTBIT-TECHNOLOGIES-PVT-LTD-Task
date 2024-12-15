# QUANTBIT-TECHNOLOGIES-PVT-LTD-Task
# Image Classification: MNIST Handwritten Digit Classification using CNN

This project focuses on classifying images of handwritten digits (0-9) using the **MNIST dataset**. The classification is performed using **Convolutional Neural Networks (CNNs)** implemented in TensorFlow/Keras. The project includes data preprocessing, model creation, training, and evaluation.

# Table of Contents  
1. [Overview](#overview)  
2. [Dataset Description](#dataset-description)  
3. [Technologies Used](#technologies-used)  
4. [Project Workflow](#project-workflow)  
5. [Model Architecture](#model-architecture)  
6. [Results](#results)  
7. [How to Run](#how-to-run)  
8. [File Structure](#file-structure)  
9. [References](#references)


# Overview  
The goal of this project is to develop a deep learning model to classify handwritten digits using the *MNIST dataset*. The dataset contains grayscale images (28x28 pixels) of digits from 0 to 9.  
By utilizing **Convolutional Neural Networks (CNNs)*, we achieve high accuracy in predicting the digits.

# Dataset Description  
- **Name**: MNIST (Modified National Institute of Standards and Technology) Dataset  
- **Size**: 70,000 images (60,000 training and 10,000 testing images)  
- **Image Details**:  
  - Grayscale images  
  - Resolution: 28x28 pixels  
  - Classes: Digits from 0 to 9  

For more information about the dataset: [TensorFlow MNIST Dataset](https://www.tensorflow.org/datasets/catalog/mnist)  

## Technologies Used  

- **Python 3.x**  
- **TensorFlow/Keras** - For building and training the CNN model  
- **NumPy** - For numerical operations  
- **Matplotlib** - For data visualization  
- **Jupyter Notebook** - For development and testing  

## Project Workflow  

1. Data Loading and Preprocessing:  
   - Load MNIST dataset using TensorFlow.  
   - Reshape images to a shape suitable for CNN input (`28x28x1`).  
   - Normalize pixel values to range `[0, 1]`.  
   - Perform one-hot encoding for labels.

2. Model Building:  
   - Define a CNN architecture with Conv2D, MaxPooling2D, and Dense layers.  
   - Use Softmax activation for output to predict probabilities.

3. **Model Training**:  
   - Optimizer: Adam  
   - Loss Function: Categorical Crossentropy  
   - Train model with `10 epochs` and `batch size = 64`.

4. **Evaluation**:  
   - Evaluate the model on the test set.  
   - Visualize predictions and model performance.

5. **Visualization**:  
   - Plot sample test images alongside predicted and true labels.

6. **Visualizing a Random Subset of MNIST Images with Their Labels**:  
   - Select a random subset of 16 images from the training dataset.  
   - Display these images along with their true labels in a 4x4 grid for easy viewing.

7. **Plotting the Accuracy Graph**:  
   - Simulate the training and validation accuracy across epochs.  
   - Plot the accuracy trends to visualize how the model improves over time during training.

## Model Architecture  

| **Layer**             | **Output Shape**   | **Details**                     |  
|-----------------------|--------------------|---------------------------------|  
| Input                 | (28, 28, 1)       | Grayscale image input            |  
| Conv2D (1st)          | (26, 26, 32)      | 32 filters, kernel size (3x3)    |  
| MaxPooling2D          | (13, 13, 32)      | Pool size (2x2)                  |  
| Conv2D (2nd)          | (11, 11, 64)      | 64 filters, kernel size (3x3)    |  
| MaxPooling2D          | (5, 5, 64)        | Pool size (2x2)                  |  
| Flatten               | (1600)            | Flattened feature maps           |  
| Dense (Hidden)        | (128)             | Fully connected layer            |  
| Dense (Output)        | (10)              | Softmax activation for 10 classes|  

## Results  

- **Training Accuracy**: ~99%  
- **Test Accuracy**: ~98-99%  
- Visualized predictions confirm that the model classifies digits accurately.  

## How to Run  

1. Clone the repository:  
   git clone <repository-link>
   cd <repository-folder>

2. Install dependencies:  
   ```
   pip install tensorflow matplotlib numpy
   ```

3. Run the notebook:  
   ```
   jupyter notebook Ankita_project.ipynb
   ```

4. Execute all cells in the notebook to train and evaluate the model.

## File Structure  


mnist-digit-classification/
│
├── Ankita_project.ipynb       # Main project notebook
├── README.md                  # Project description
└── requirements.txt           # List of required libraries

## References  

1. [TensorFlow MNIST Dataset](https://www.tensorflow.org/datasets/catalog/mnist)  
2. [Keras Documentation](https://keras.io/)  

## Author  
**Ankita Jadhav**  
**Contact**: ankitajadhav20004@gmail.com  

