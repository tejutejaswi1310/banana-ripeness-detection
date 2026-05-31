import joblib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Dataset path
DATASET_PATH = 'dataset'

# Image size
IMG_SIZE = 128

# LBP parameters
radius = 3
n_points = 8 * radius

# Feature lists
color_features = []
combined_features = []
labels = []

# Function to extract RGB histogram features
def extract_color_features(image):

    hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])

    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()

    # Normalize histogram
    hist = hist / np.sum(hist)

    return hist


# Function to extract texture features using LBP
def extract_texture_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)

    return hist


# Load dataset
for category in os.listdir(DATASET_PATH):

    category_path = os.path.join(DATASET_PATH, category)

    for image_name in os.listdir(category_path):

        image_path = os.path.join(category_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract features
        color_feat = extract_color_features(image)
        texture_feat = extract_texture_features(image)

        combined_feat = np.concatenate(
            [color_feat, texture_feat]
        )

        # Store features
        color_features.append(color_feat)
        combined_features.append(combined_feat)
        labels.append(category)

# Convert to numpy arrays
color_features = np.array(color_features)
combined_features = np.array(combined_features)
labels = np.array(labels)

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Split dataset
X_train_c, X_test_c, y_train, y_test = train_test_split(
    color_features,
    encoded_labels,
    test_size=0.2,
    random_state=42
)

X_train_comb, X_test_comb, _, _ = train_test_split(
    combined_features,
    encoded_labels,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# MODEL 1 — COLOR FEATURES ONLY
# -----------------------------------

print("\nTraining SVM using Color Features...\n")

model_color = SVC(kernel='linear')

model_color.fit(X_train_c, y_train)

pred_color = model_color.predict(X_test_c)

print("Color Feature Accuracy:")
print(accuracy_score(y_test, pred_color))

print("\nClassification Report (Color Features):\n")

print(classification_report(
    y_test,
    pred_color,
    target_names=encoder.classes_
))

# -----------------------------------
# MODEL 2 — COLOR + TEXTURE FEATURES
# -----------------------------------

print("\nTraining SVM using Color + Texture Features...\n")

model_combined = SVC(kernel='linear')

model_combined.fit(X_train_comb, y_train)

# Save model
joblib.dump(model_combined, "banana_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("Model saved successfully!")


pred_combined = model_combined.predict(X_test_comb)

print("Combined Feature Accuracy:")
print(accuracy_score(y_test, pred_combined))

print("\nClassification Report (Combined Features):\n")

print(classification_report(
    y_test,
    pred_combined,
    target_names=encoder.classes_
))

# Confusion Matrix
cm = confusion_matrix(y_test, pred_combined)

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap='Blues')

plt.title('Confusion Matrix')
plt.colorbar()

classes = encoder.classes_

plt.xticks(np.arange(len(classes)), classes)
plt.yticks(np.arange(len(classes)), classes)

plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(len(classes)):
    for j in range(len(classes)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center',
            color='black'
        )

plt.show()