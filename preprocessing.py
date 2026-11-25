import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset path
dataset_path = "outputs/synthetic"

classes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

images = []
labels = []

print("\n===== CERVISCAN AI - IMAGE PREPROCESSING =====\n")

# Load dataset
for class_name in classes:

    folder_path = os.path.join(dataset_path, class_name)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    count = 0

    for file in os.listdir(folder_path):

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(folder_path, file)

            image = cv2.imread(image_path)

            if image is None:
                continue

            # Resize image
            image = cv2.resize(image, (224, 224))

            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Normalize pixel values
            image = image.astype("float32") / 255.0

            images.append(image)
            labels.append(class_name)

            count += 1

    print(f"{class_name}: {count} images loaded")


# Convert into NumPy arrays
X = np.array(images, dtype=np.float32)
y = np.array(labels)

# Add channel dimension for CNN
X = np.expand_dims(X, axis=-1)

print("\nTotal Images:", len(X))
print("Image Shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== DATASET SPLITTING =====")

print("Training Images:", len(X_train))
print("Testing Images:", len(X_test))

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Save preprocessed data
output_path = "outputs/preprocessed"

os.makedirs(output_path, exist_ok=True)

np.save(os.path.join(output_path, "X_train.npy"), X_train)
np.save(os.path.join(output_path, "X_test.npy"), X_test)

np.save(os.path.join(output_path, "y_train.npy"), y_train)
np.save(os.path.join(output_path, "y_test.npy"), y_test)

print("\nPreprocessed datasets saved successfully!")

print("\n===== PREPROCESSING COMPLETED =====")