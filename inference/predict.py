import numpy as np
from tensorflow import keras  # Using Keras (TensorFlow) for model loading and prediction

# Paths to the trained model files (update these paths if needed)
HORIZON_MODEL_PATH = "models/horizon_model.h5"
STARS_MODEL_PATH   = "models/stars_model.h5"
QUALITY_MODEL_PATH = "models/quality_model.h5"

# Global model variables (loaded on first use)
_horizon_model = None
_stars_model   = None
_quality_model = None

def _load_models():
    """Internal helper to load models into memory if not already loaded."""
    global _horizon_model, _stars_model, _quality_model
    if _horizon_model is None:
        _horizon_model = keras.models.load_model(HORIZON_MODEL_PATH)
    if _stars_model is None:
        _stars_model = keras.models.load_model(STARS_MODEL_PATH)
    if _quality_model is None:
        _quality_model = keras.models.load_model(QUALITY_MODEL_PATH)

def classify_image(image_path):
    """
    Run inference on the image at `image_path` using all three models.
    Returns a combined label string like "HORIZON, STARS, GOOD".
    """
    # Load models if not already loaded
    _load_models()
    # Load and preprocess the image
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0  # normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)    # shape (1, H, W, C)

    # Model 1: Horizon detection
    pred1 = _horizon_model.predict(img_array)
    # Determine label for horizon model (assume binary classification)
    # If model output is a probability or single sigmoid output:
    if pred1.shape[-1] == 1:
        horizon_prob = float(pred1[0])
        label_horizon = "HORIZON" if horizon_prob >= 0.5 else "NO_HORIZON"
    else:
        # If model output is one-hot or softmax for two classes
        class_idx = int(np.argmax(pred1[0]))
        label_horizon = "HORIZON" if class_idx == 1 else "NO_HORIZON"

    # Model 2: Star detection
    pred2 = _stars_model.predict(img_array)
    if pred2.shape[-1] == 1:
        star_prob = float(pred2[0])
        label_stars = "STARS" if star_prob >= 0.5 else "NO_STARS"
    else:
        class_idx = int(np.argmax(pred2[0]))
        label_stars = "STARS" if class_idx == 1 else "NO_STARS"

    # Model 3: Image quality (good vs bad)
    pred3 = _quality_model.predict(img_array)
    if pred3.shape[-1] == 1:
        quality_prob = float(pred3[0])
        label_quality = "GOOD" if quality_prob >= 0.5 else "BAD"
    else:
        class_idx = int(np.argmax(pred3[0]))
        label_quality = "GOOD" if class_idx == 1 else "BAD"

    # Combine results from all models into a single label string
    combined_label = f"{label_horizon}, {label_stars}, {label_quality}"
    return combined_label
