import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


dataset_file = sys.argv[1]
mode = sys.argv[2]

if mode not in ['train', 'test']:
    print("Mode must be 'train' or 'test'")
    sys.exit(1)

print(f"Loading data from: {dataset_file}")

# Load data from .npz file
data = np.load(dataset_file)
X = data['images']
y = data['labels']

# print(f"Data loaded: X shape = {X.shape}, y shape = {y.shape}")
# print(f"Number of classes: {len(np.unique(y))}")

# calling the stratify method to ensures we keep the same ratio of t-shirts/shoes in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


input_shape = X_train.shape[1:] # (incase we decide to change the shape passed to the sys.argv[])
label_classes = len(np.unique(y))  

# print(f"Number of classes: {label_classes}")
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# NOTE ---------------------------- Start  working from the code below ---------------------------- #

# --- 5. Main Logic Flow ---
MODEL_PATH = "my_model.keras" # [cite: 36, 41]

if mode == 'train':
    # Define Model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(label_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    print("Starting training...")
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    # Save
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

elif mode == 'test':
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train first!")
        sys.exit(1)

    print("Loading model...")
    model = models.load_model(MODEL_PATH)
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # Visualizing the Confusion Matrix is a requirement [cite: 41]
    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()