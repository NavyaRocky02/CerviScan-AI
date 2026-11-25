import numpy as np

print("\n===== CERVISCAN AI DATASET CHECK =====")

path = "outputs/preprocessed"

X_train = np.load(path + "/X_train.npy")
X_test = np.load(path + "/X_test.npy")

y_train = np.load(path + "/y_train.npy")
y_test = np.load(path + "/y_test.npy")

print("\nImage Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\nTraining Labels:")

labels, counts = np.unique(y_train, return_counts=True)

for label, count in zip(labels, counts):
    print(label, ":", count)

print("\nTesting Labels:")

labels, counts = np.unique(y_test, return_counts=True)

for label, count in zip(labels, counts):
    print(label, ":", count)

print("\nPixel Values:")
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())
print("Mean:", X_train.mean())
print("Standard Deviation:", X_train.std())

print("\nClass-wise Image Statistics:")

for label in np.unique(y_train):

    images = X_train[y_train == label]

    print(
        label,
        "Images:", len(images),
        "Mean:", round(float(images.mean()), 3),
        "Std:", round(float(images.std()), 3)
    )

print("\n===== DATASET CHECK COMPLETED =====")