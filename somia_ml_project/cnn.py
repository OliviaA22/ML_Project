import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

npz_file = sys.argv[1]
mode = sys.argv[2]

data = np.load(npz_file)
X = data["X"]
T = data["T"]

num_classes = len(np.unique(T))
input_shape = X.shape[1:]


# --------------------------------------------------
# DNN Model
# --------------------------------------------------
def build_dnn():
    model = Sequential(
        [
            Flatten(input_shape=input_shape),
            Dense(128, activation="relu", kernel_regularizer=L2(0.001)),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=L2(0.001)),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------
# CNN Model (LeNet-5 inspired)
# --------------------------------------------------
def build_cnn():
    model = Sequential(
        [
            Conv2D(
                6,
                kernel_size=5,
                activation="relu",
                input_shape=input_shape,
                kernel_regularizer=L2(0.001),
            ),
            AveragePooling2D(pool_size=2),
            Dropout(0.25),
            Conv2D(16, kernel_size=5, activation="relu", kernel_regularizer=L2(0.001)),
            AveragePooling2D(pool_size=2),
            Dropout(0.25),
            Flatten(),
            Dense(120, activation="relu", kernel_regularizer=L2(0.001)),
            Dropout(0.3),
            Dense(84, activation="relu", kernel_regularizer=L2(0.001)),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------
# TRAIN MODE
# --------------------------------------------------
if mode == "train":
    # Fix class imbalance by oversampling minority class (class 5)
    print("\n" + "=" * 50)
    print("Handling class imbalance...")
    print("=" * 50)

    X_balanced = list(X)
    T_balanced = list(T)

    unique, counts = np.unique(T, return_counts=True)
    max_count = np.max(counts)
    min_class = unique[np.argmin(counts)]
    min_count = np.min(counts)

    print(f"Min class: {int(min_class)} with {min_count} samples")
    print(f"Max class: {int(unique[np.argmax(counts)])} with {max_count} samples")

    # Oversample minority class
    minority_indices = np.where(T == min_class)[0]
    oversample_count = max_count - min_count
    oversample_indices = np.random.choice(
        minority_indices, size=oversample_count, replace=True
    )

    X_balanced.extend(X[oversample_indices])
    T_balanced.extend(T[oversample_indices])

    X_balanced = np.array(X_balanced)
    T_balanced = np.array(T_balanced)

    print(f"Balanced dataset: {X_balanced.shape[0]} samples")
    print("=" * 50)

    X_train, X_val, y_train, y_val = train_test_split(
        X_balanced, T_balanced, test_size=0.2, random_state=42
    )

    print("\n" + "=" * 50)
    print("Training DNN...")
    print("=" * 50)
    dnn = build_dnn()

    early_stop_dnn = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    reduce_lr_dnn = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    )

    dnn.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop_dnn, reduce_lr_dnn],
        verbose=1,
    )
    dnn.save("dnn_model.keras")

    print("\nEvaluating DNN on validation set...")
    loss_dnn, acc_dnn = dnn.evaluate(X_val, y_val, verbose=0)
    print(f"DNN Validation Accuracy: {acc_dnn:.4f}")

    pred_dnn = dnn.predict(X_val, verbose=0)
    y_pred_dnn = np.argmax(pred_dnn, axis=1)
    cm_dnn = confusion_matrix(y_val, y_pred_dnn)

    print("\n" + "=" * 50)
    print("Training CNN...")
    print("=" * 50)
    cnn = build_cnn()

    early_stop_cnn = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    reduce_lr_cnn = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    )

    cnn.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop_cnn, reduce_lr_cnn],
        verbose=1,
    )
    cnn.save("cnn_model.keras")

    print("\nEvaluating CNN on validation set...")
    loss_cnn, acc_cnn = cnn.evaluate(X_val, y_val, verbose=0)
    print(f"CNN Validation Accuracy: {acc_cnn:.4f}")

    pred_cnn = cnn.predict(X_val, verbose=0)
    y_pred_cnn = np.argmax(pred_cnn, axis=1)
    cm_cnn = confusion_matrix(y_val, y_pred_cnn)

    # Display confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    disp_dnn = ConfusionMatrixDisplay(confusion_matrix=cm_dnn, display_labels=None)
    disp_dnn.plot(ax=ax[0], cmap="Blues", values_format="d")
    ax[0].set_title(f"DNN Model (Acc: {acc_dnn:.2%})")

    disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=None)
    disp_cnn.plot(ax=ax[1], cmap="Blues", values_format="d")
    ax[1].set_title(f"CNN Model (Acc: {acc_cnn:.2%})")

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("Training complete! Models saved.")
    print("=" * 50)

# --------------------------------------------------
# TEST MODE (CLEAN DATA ONLY)
# --------------------------------------------------
else:
    from tensorflow.keras.models import load_model

    print("\n" + "=" * 50)
    print("Testing on clean data...")
    print("=" * 50)

    # Load DNN
    dnn = load_model("dnn_model.keras")
    print("\nEvaluating DNN...")
    pred_dnn = dnn.predict(X, batch_size=256, verbose=1)
    y_pred_dnn = np.argmax(pred_dnn, axis=1)
    acc_dnn = np.mean(y_pred_dnn == T)
    print(f"DNN Test Accuracy: {acc_dnn:.4f}")

    cm_dnn = confusion_matrix(T, y_pred_dnn)

    # Load CNN
    cnn = load_model("cnn_model.keras")
    print("\nEvaluating CNN...")
    pred_cnn = cnn.predict(X, batch_size=256, verbose=1)
    y_pred_cnn = np.argmax(pred_cnn, axis=1)
    acc_cnn = np.mean(y_pred_cnn == T)
    print(f"CNN Test Accuracy: {acc_cnn:.4f}")

    cm_cnn = confusion_matrix(T, y_pred_cnn)

    # Display confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    disp_dnn = ConfusionMatrixDisplay(confusion_matrix=cm_dnn, display_labels=None)
    disp_dnn.plot(ax=ax[0], cmap="Blues", values_format="d")
    ax[0].set_title(f"DNN Model (Acc: {acc_dnn:.2%})")

    disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=None)
    disp_cnn.plot(ax=ax[1], cmap="Blues", values_format="d")
    ax[1].set_title(f"CNN Model (Acc: {acc_cnn:.2%})")

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"DNN Accuracy: {acc_dnn:.4f}")
    print(f"CNN Accuracy: {acc_cnn:.4f}")
    print("=" * 50)
