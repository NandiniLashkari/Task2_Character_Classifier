import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import zipfile
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from google.colab import files

# Step 1: Upload the dataset
uploaded = files.upload()  # Upload 'dataset.zip'
zip_path = "/content/dataset.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/unzipped_folder")

# Step 2: Load the CSV and update image paths
df = pd.read_csv("/content/unzipped_folder/task2_Dataset_All/all_labels.csv")
image_folder = "/content/unzipped_folder/task2_Dataset_All"
df['image'] = df['image'].apply(lambda x: os.path.join(image_folder, x))

# Debug: Check if files exist
print(f"First image path: {df['image'][0]}")
print(f"File exists: {os.path.exists(df['image'][0])}")
print(f"Files in folder: {os.listdir(image_folder)[:5]}")

# Step 3: Function to preprocess images
def img_to_arr(x):
    img = cv2.imread(x)
    if img is None:
        print(f"Failed to load image: {x}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    return img

# Step 4: Load and preprocess images
images = []
for img_path in df['image']:
    img = img_to_arr(img_path)
    if img is not None:
        images.append(img)

# Filter out failed images
failed_images = [img_path for img_path, img in zip(df['image'], images) if img is None]
images = [img for img in images if img is not None]
print(f"Successfully loaded images: {len(images)}")
print(f"Failed images: {failed_images}")

# Update DataFrame to match successfully loaded images
success_mask = [img is not None for img in df['image'].map(img_to_arr)]
df = df[success_mask].reset_index(drop=True)

# Step 5: Prepare data for inference
x = np.array(images).reshape(-1, 64, 64, 1)
x_normalized = x / 255.0  # Normalize
y = df['label']
le = LabelEncoder()
y_label = le.fit_transform(y)

# Step 6: Upload and load the pre-trained model
uploaded = files.upload()  # Upload 'digit_classifier_cnn.h5'
model = tf.keras.models.load_model('/content/digit_classifier_cnn.h5')
print("Model loaded successfully from 'digit_classifier_cnn.h5'")

# Step 7: Perform inference
y_pred = model.predict(x_normalized)
y_pred_labels = np.argmax(y_pred, axis=1)

# Step 8: Evaluate predictions
accuracy = accuracy_score(y_label, y_pred_labels)
precision = precision_score(y_label, y_pred_labels, average='weighted')
recall = recall_score(y_label, y_pred_labels, average='weighted')
f1 = f1_score(y_label, y_pred_labels, average='weighted')
confusion_mat = confusion_matrix(y_label, y_pred_labels)

print("Inference Results on Dataset:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)

# Step 9: Identify the top 3 most confused character pairs
misclassifications = []
for true_idx in range(len(confusion_mat)):
    for pred_idx in range(len(confusion_mat)):
        if true_idx != pred_idx and confusion_mat[true_idx][pred_idx] > 0:
            true_char = le.inverse_transform([true_idx])[0]
            pred_char = le.inverse_transform([pred_idx])[0]
            count = confusion_mat[true_idx][pred_idx]
            misclassifications.append((true_char, pred_char, count))

misclassifications.sort(key=lambda x: x[2], reverse=True)
print("\nTop 3 most confused character pairs (True, Predicted, Count):")
for i, (true_char, pred_char, count) in enumerate(misclassifications[:3], 1):
    print(f"{i}. ({true_char}, {pred_char}): {count} times")