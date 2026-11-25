import os
import matplotlib.pyplot as plt

dataset_path = "outputs/synthetic"

classes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

image_counts = []

for class_name in classes:

    folder_path = os.path.join(dataset_path, class_name)

    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    image_counts.append(len(images))


# Bar Chart

plt.figure(figsize=(10, 5))

plt.bar(classes, image_counts)

plt.title("CerviScan AI - Dataset Distribution")

plt.xlabel("Cervical Classes")

plt.ylabel("Number of Images")

plt.savefig("dataset_distribution.png")

plt.show()


# Pie Chart

plt.figure(figsize=(8, 8))

plt.pie(
    image_counts,
    labels=classes,
    autopct="%1.1f%%"
)

plt.title("Dataset Class Distribution")

plt.savefig("class_distribution.png")

plt.show()