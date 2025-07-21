import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tensorflow import keras
# Import our modules
import data_utils
import model_utils

def evaluate_and_print(model, dataset, subset_name):
    """Compute and print accuracy, precision, recall, and confusion matrix for the model on a dataset."""
    # Get true labels and predictions
    y_true = []
    y_pred = []
    for batch_images, batch_labels in dataset:
        # Predict probabilities for each batch
        preds = model.predict(batch_images, verbose=0)
        # Convert probabilities to binary class predictions (threshold 0.5)
        batch_preds = (preds.reshape(-1) >= 0.5).astype(int)
        y_true.extend(batch_labels.numpy().astype(int).tolist())
        y_pred.extend(batch_preds.tolist())
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nEvaluation on {subset_name}:")
    print(f"Accuracy = {acc*100:.2f}%")
    print(f"Precision = {prec*100:.2f}%")
    print(f"Recall    = {rec*100:.2f}%")
    print("Confusion Matrix (actual vs predicted):")
    print(cm)  # This will be a 2x2 matrix [[TN, FP],[FN, TP]]

def main():
    # Paths
    data_dir = os.path.join("models_training", "data")
    os.makedirs("output_models", exist_ok=True)
    # Train and evaluate for each category: 'horizon' and 'stars'
    for category in ["stars", "horizon"]:
        print(f"\n=== Training model for category: {category} ===")
        # Load datasets
        train_ds, val_ds, test_ds = data_utils.get_datasets_for_category(data_dir, category)
        # Build model
        model = model_utils.build_model(input_shape=(data_utils.IMG_HEIGHT, data_utils.IMG_WIDTH, 3))
        # Train model
        model, history = model_utils.train_model(model, train_ds, val_ds, category)
        # Evaluate on validation and test sets
        evaluate_and_print(model, val_ds, f"{category} validation set")
        evaluate_and_print(model, test_ds, f"{category} test set")
        # Convert model to TFLite and save
        converter = keras.models.save_model  # placeholder to avoid linter issues (reassigned below)
        try:
            # Use the TFLiteConverter API to convert the model:contentReference[oaicite:15]{index=15}
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            tflite_path = os.path.join("output_models", f"{category}_model.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            print(f"Saved TFLite model for '{category}' at: {tflite_path}")
        except Exception as e:
            print(f"Error converting {category} model to TFLite: {e}")

if __name__ == "__main__":
    import tensorflow as tf  # placed inside to ensure TF is loaded in main context
    main()

