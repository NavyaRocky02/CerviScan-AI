import os
from PIL import Image

dataset_path = "outputs/synthetic"

classes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

total_images = 0

print("\nCERVISCAN AI - DATASET ANALYSIS\n")

for class_name in classes:

    folder_path = os.path.join(dataset_path, class_name)

    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    count = len(images)
    total_images += count

    print(f"{class_name}: {count} images")

print("\nTotal Images:", total_images)

print("\nImage Dimensions:")

for class_name in classes:

    folder_path = os.path.join(dataset_path, class_name)

    images = os.listdir(folder_path)

    if images:
        image_path = os.path.join(folder_path, images[0])

        image = Image.open(image_path)

        print(f"{class_name}: {image.size}")