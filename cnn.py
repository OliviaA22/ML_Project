import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


npz_file = sys.argv[1]
mode = sys.argv[2]  # 'train' or 'test'

data = np.load(npz_file)
X = data["images"]
y = data["labels"]

data = np.load(npz_file)
X = data["images"]
y = data["labels"]

X = X.astype("float32") / 255.0

num_classes = len(np.unique(y))
input_shape = X.shape[1:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def build_dnn(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"])
    return model

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(6, kernel_size=5, activation="relu"),
        MaxPooling2D(),
        Conv2D(16, kernel_size=5, activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(120, activation="relu"),
        Dense(84, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"])
    return model




if mode == "train":
    print(f"Training model on {len(X_train)} samples")
    
    dnn = build_dnn(input_shape, num_classes)
    dnn.fit(X_train, y_train, epochs=5,  batch_size=64, validation_split=0.1)
    dnn.save("dnn_model.keras") 
    # dnn.save_weights("dnn.weights.h5")

    cnn = build_cnn(input_shape, num_classes)
    cnn.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    cnn.save("cnn_model.keras")
    # cnn.save_weights("cnn.weights.h5")

else:
    print("Loading models...")
    dnn = load_model("dnn_model.keras")
    # dnn.load_weights("dnn.weights.h5")
    cnn = load_model("cnn_model.keras")
    # cnn.load_weights("cnn.weights.h5")

    loss_dnn, acc_dnn = dnn.evaluate(X_test, y_test, verbose=0)
    print(f"DNN Test Accuracy: {acc_dnn:.4f}")
    
    pred_dnn = dnn.predict(X_test)
    cm_dnn = confusion_matrix(y_test, np.argmax(pred_dnn, axis=1))


    loss_cnn, acc_cnn = cnn.evaluate(X_test, y_test, verbose=0)
    print(f"CNN Test Accuracy: {acc_cnn:.4f}")
    
    pred_cnn = cnn.predict(X_test)
    cm_cnn = confusion_matrix(y_test, np.argmax(pred_cnn, axis=1))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    disp_dnn = ConfusionMatrixDisplay(confusion_matrix=cm_dnn, display_labels=None)
    disp_dnn.plot(ax=ax[0], cmap='Blues', values_format='d')
    ax[0].set_title(f"DNN Model (Acc: {acc_dnn:.2%})")


    disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=None)
    disp_cnn.plot(ax=ax[1], cmap='Blues', values_format='d')
    ax[1].set_title(f"CNN Model (Acc: {acc_cnn:.2%})")


    plt.tight_layout()
    plt.show()