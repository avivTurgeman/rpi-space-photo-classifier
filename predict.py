import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained models (these files should exist from the training phase)
model_stars = tf.keras.models.load_model("stars_model.h5")
model_horizon = tf.keras.models.load_model("horizon_model.h5")

# Define image size expected by the models (for example, 224x224)
IMG_SIZE = (224, 224)

def predict_stars(image: np.ndarray) -> bool:
    """Return True if the image is classified as a stars image, False otherwise."""
    # Preprocess the image for the stars model
    img_resized = cv2.resize(image, IMG_SIZE)
    img_array = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)
    # Model prediction (assuming binary classification: output > 0.5 means 'stars')
    pred = model_stars.predict(img_array)[0][0]
    return pred > 0.5

def predict_horizon(image: np.ndarray) -> bool:
    """Return True if the image is classified as an horizon image, False otherwise."""
    # Preprocess the image for the horizon model
    img_resized = cv2.resize(image, IMG_SIZE)
    img_array = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)
    # Model prediction (binary classification for horizon vs not-horizon)
    pred = model_horizon.predict(img_array)[0][0]
    return pred > 0.5
