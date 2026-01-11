# Fashion MNIST Classification - ML Project

A deep learning project implementing CNN and DNN models to classify Fashion MNIST images with 90%+ accuracy on validation data.

## Project Overview

This project trains two neural network models (DNN and CNN) on Fashion MNIST dataset with the following results:

- **Validation Accuracy (DNN):** 91.37% ✅
- **Validation Accuracy (CNN):** 90.79% ✅
- **Test Accuracy (DNN):** 78.72%
- **Test Accuracy (CNN):** 79.90%

> All models exceed the 88% accuracy requirement on validation data.

## Dataset

- **Source:** Fashion MNIST (28×28 grayscale images, 10 classes)
- **Training Data:** 54,197 images (after data cleaning and balancing)
- **Test Data:** 9,999 images
- **Classes:** T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot

## Project Structure

```
├── convert_data.py          # Image preprocessing pipeline
├── cnn.py                   # Model training and evaluation
├── .gitignore               # Exclude large dataset files
└── README.md                # This file
```

## Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/OliviaA22/ML_Project.git
cd ML_Project
git checkout somia-ml-project
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy scikit-learn pillow matplotlib
```

### 3. Prepare Dataset

Download Fashion MNIST datasets:
- **Training data:** [data1.zip](https://link-to-data1)
- **Test data:** [data2.zip](https://link-to-data2)

Extract to parent directory:
```
ML_Project/
├── fashion_mnist/          # Extract data1.zip here
├── test_fashion_mnist/     # Extract data2.zip here
└── somia_ml_project/       # This folder
```

## Usage

### Step 1: Generate Training Dataset
```bash
python convert_data.py "..\fashion_mnist" 28 28 1 "train.npz" 1
```

**Arguments:**
- `input_path`: Path to training images directory
- `width`: Image width (28)
- `height`: Image height (28)
- `channels`: 1 for grayscale, 3 for RGB
- `output_file`: Output NPZ filename
- `correct_data`: 1 to remove corrupted images, 0 to keep all

### Step 2: Train Models
```bash
python cnn.py train.npz train
```

This trains both DNN and CNN models:
- Saves `dnn_model.keras`
- Saves `cnn_model.keras`
- Displays validation accuracy and training curves

### Step 3: Test Models
```bash
python cnn.py test.npz test
```

Evaluates both models on test set:
- Displays test accuracy for each model
- Generates confusion matrices
- Shows detailed classification metrics

## Model Architectures

### DNN (Dense Neural Network)
```
Flatten(28×28×1)
  ↓
Dense(128, ReLU) + Dropout(0.3) + L2(0.001)
  ↓
Dense(64, ReLU) + Dropout(0.2) + L2(0.001)
  ↓
Dense(10, Softmax)
```

### CNN (LeNet-5 Inspired)
```
Conv2D(6, 5×5, ReLU) + L2(0.001)
  ↓
AveragePooling2D(2×2) + Dropout(0.25)
  ↓
Conv2D(16, 5×5, ReLU) + L2(0.001)
  ↓
AveragePooling2D(2×2) + Dropout(0.25)
  ↓
Flatten
  ↓
Dense(120, ReLU) + Dropout(0.3) + L2(0.001)
  ↓
Dense(84, ReLU) + Dropout(0.2) + L2(0.001)
  ↓
Dense(10, Softmax)
```

## Key Features

### Data Preprocessing (convert_data.py)
- **Image Loading:** Reads PNG images and extracts labels from filenames
- **Corruption Detection:** Removes blank images (std < 5) and extreme values
- **Normalization:** Scales pixel values to [0, 1]
- **Analysis:** Prints class distribution and data statistics

### Model Training (cnn.py)
- **Class Balancing:** Oversamples minority class (Class 5) to prevent bias
- **Regularization:** L2 (0.001) + Dropout (0.2-0.3) to reduce overfitting
- **Callbacks:**
  - Early Stopping (patience=3): Stops training if validation loss doesn't improve
  - ReduceLROnPlateau: Reduces learning rate by 50% if stuck
- **Batch Size:** 64
- **Learning Rate:** 0.001 (decaying)
- **Optimizer:** Adam

### Model Evaluation
- **Confusion Matrices:** Visual classification performance per class
- **Accuracy Metrics:** Overall and per-class accuracy
- **Batch Prediction:** Efficient inference on test set

## Results & Analysis

### Training Performance
- Both models successfully trained for 15 epochs maximum
- Early stopping prevented overfitting
- Validation accuracy stabilized at ~91% (DNN) and ~91% (CNN)

### Test Performance
- Test accuracy lower than validation due to distribution mismatch
  - Training data (data1.zip) and test data (data2.zip) have different characteristics
  - This is expected real-world behavior
- CNN slightly outperforms DNN on test set (79.90% vs 78.72%)

### Key Improvements Implemented
1. **Class Imbalance Fix:** Oversampled minority class from 199 → 5,801 samples (+3.34% accuracy)
2. **Regularization:** L2 + Dropout reduced overfitting significantly
3. **Learning Rate Scheduling:** Adaptive learning rate improved convergence
4. **Data Cleaning:** Removed corrupted images (3 images with extreme values)

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy
- scikit-learn
- Pillow
- Matplotlib

## Troubleshooting

**Issue:** Module not found
```bash
pip install --upgrade tensorflow scikit-learn
```

**Issue:** Out of memory
- Reduce `batch_size` in `cnn.py` (line with `batch_size=64`)
- Reduce number of samples during training

**Issue:** Low accuracy
- Ensure data is properly preprocessed with `convert_data.py`
- Check class distribution in converted NPZ files
- Verify image dimensions are 28×28

## Author

Somia - ML Project 1 (Alexander Gepperth's Course)

## License

This project is part of a university course assignment.

---

**Last Updated:** January 11, 2026
**Status:** ✅ Project Complete - All Requirements Met
