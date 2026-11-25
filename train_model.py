import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("\n===== CERVISCAN AI - IMPROVED CNN TRAINING =====\n")

# 1. Load dataset
data_path = "outputs/preprocessed"

X_train = np.load(os.path.join(data_path, "X_train.npy"))
X_test = np.load(os.path.join(data_path, "X_test.npy"))

y_train = np.load(os.path.join(data_path, "y_train.npy"))
y_test = np.load(os.path.join(data_path, "y_test.npy"))

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# 2. Prepare images
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

if X_train.max() > 1:
    X_train /= 255.0
    X_test /= 255.0

if X_train.ndim == 3:
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

# 3. Prepare labels
classes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

if set(np.unique(np.concatenate([y_train, y_test]))) != set(classes):
    raise ValueError("Dataset labels do not match C1 to C7.")

label_map = {label: index for index, label in enumerate(classes)}

y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

# 4. Create a stratified validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nTraining images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))

# 5. Build a smaller CNN
model = Sequential([
    tf.keras.Input(shape=(224, 224, 1)),

    Conv2D(16, 3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(32, 3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, 3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    GlobalAveragePooling2D(),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(7, activation="softmax")
])

# 6. Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 7. Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 8. Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# 9. Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=16,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# 10. Evaluate
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===== MODEL EVALUATION =====")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# 11. Predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    predicted_labels,
    labels=list(range(7)),
    target_names=classes,
    zero_division=0
))

# 12. Confusion matrix
cm = confusion_matrix(
    y_test,
    predicted_labels,
    labels=list(range(7))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(cmap="Blues")
plt.title("CerviScan AI - Confusion Matrix")
plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# 13. Training graph
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/training_performance.png", dpi=300, bbox_inches="tight")
plt.close()

# 14. Save model
model.save("models/cerviscan_cnn.keras")

# 15. Save class names
with open("models/class_names.json", "w") as file:
    json.dump(classes, file, indent=4)

# 16. Save evaluation results
results = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "classes": classes,
    "dataset_type": "Synthetic"
}

with open("results/evaluation_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\n===== TRAINING COMPLETED =====")
print("Model saved in models/cerviscan_cnn.keras")
print("Best model saved in models/best_model.keras")
print("Evaluation results saved in results/")