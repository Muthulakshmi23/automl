# src/preprocess.py
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, OneHotEncoder

IMG_SIZE = 224

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return img

def load_data(image_dir, metadata_path):
    tabular_data = pd.read_csv(metadata_path)
    tabular_data.fillna(tabular_data.mean(), inplace=True)

    images = []
    labels = []
    for idx, row in tabular_data.iterrows():
        img_path = os.path.join(image_dir, row['image_id'])
        if os.path.exists(img_path):
            images.append(preprocess_image(img_path))
            labels.append(row['diagnostic'])

    # One-hot encode categorical variables
    encoder = OneHotEncoder()
    encoded_tabular = encoder.fit_transform(tabular_data[['region', 'itch', 'hurt']]).toarray()

    # Scale numerical features
    scaler = StandardScaler()
    scaled_tabular = scaler.fit_transform(tabular_data.drop(['image_id', 'diagnostic'], axis=1))

    tabular_features = np.concatenate([scaled_tabular, encoded_tabular], axis=1)
    labels_encoded = pd.get_dummies(labels).values

    return np.array(images), tabular_features, labels_encoded

# Data augmentation
def augment_images(images):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    return datagen
