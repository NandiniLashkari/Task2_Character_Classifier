import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import zipfile
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Upload and extract the dataset
from google.colab import files
uploaded = files.upload()  # Upload 'dataset.zip'
zip_path = "/content/dataset.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/unzipped_folder")

# Step 2: Load the CSV and update image paths
df = pd.read_csv("/content/unzipped_folder/task2_Dataset_All/all_labels.csv")
image_folder = "/content/unzipped_folder/task2_Dataset_All"
df['image'] = df['image'].apply(lambda x: os.path.join(image_folder, x))

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

failed_images = [img_path for img_path, img in zip(df['image'], images) if img is None]
images = [img for img in images if img is not None]
success_mask = [img is not None for img in df['image'].map(img_to_arr)]
df = df[success_mask].reset_index(drop=True)

# Step 5: Prepare data
x = np.array(images).reshape(-1, 64, 64, 1)
y = df['label']
le = LabelEncoder()
y_label = le.fit_transform(y)

# Step 6: Split and normalize data
train_images, test_images, train_labels, test_labels = train_test_split(x, y_label, test_size=0.2, random_state=42)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Step 7: Define and train the model
model = Sequential()
model.add(Conv2D(512, (5, 5), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(62, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

# Step 8: Save the trained model
model.save('/content/digit_classifier_cnn.h5')
print("Model saved as 'digit_classifier_cnn.h5' in /content/ directory")

# Step 9: Download the saved model
files.download('/content/digit_classifier_cnn.h5')
print("Model file 'digit_classifier_cnn.h5' has been downloaded to your local machine")

# Step 10: Perform inference on the entire dataset
x_normalized = x / 255.0
y_pred = model.predict(x_normalized)
y_pred_labels = np.argmax(y_pred, axis=1)

# Step 11: Evaluate predictions
accuracy = accuracy_score(y_label, y_pred_labels)
precision = precision_score(y_label, y_pred_labels, average='weighted')
recall = recall_score(y_label, y_pred_labels, average='weighted')
f1 = f1_score(y_label, y_pred_labels, average='weighted')
confusion_mat = confusion_matrix(y_label, y_pred_labels)

print("Inference Results on Entire Dataset:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)

# Step 12: Identify the top 3 most confused character pairs
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

# Step 13: If You Want to Test by Yourself
print("\n=== If You Want to Test by Yourself ===")
# Load the saved model (in case you're running this section separately)
model = tf.keras.models.load_model('/content/digit_classifier_cnn.h5')

# Select a few images from your dataset for testing (e.g., first 5 images)
num_test_images = 5
test_subset_indices = range(min(num_test_images, len(images)))
test_subset_images = x_normalized[test_subset_indices]
test_subset_labels = y_label[test_subset_indices]
test_subset_paths = df['image'].iloc[test_subset_indices].values

# Predict on the selected images
predictions = model.predict(test_subset_images)
predicted_labels = np.argmax(predictions, axis=1)

# Display the images with predicted and actual labels
plt.figure(figsize=(15, 5))
for i in range(len(test_subset_indices)):
    plt.subplot(1, num_test_images, i + 1)
    # Load the original image for display (not the preprocessed one)
    original_img = cv2.imread(test_subset_paths[i], cv2.IMREAD_GRAYSCALE)
    plt.imshow(original_img, cmap='gray')
    true_label = le.inverse_transform([test_subset_labels[i]])[0]
    pred_label = le.inverse_transform([predicted_labels[i]])[0]
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')
plt.show()
print("Displayed the first", len(test_subset_indices), "images with their true and predicted labels.")