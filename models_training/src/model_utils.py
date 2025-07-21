import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape=(224, 224, 3)):
    """Builds a binary classification model with a frozen MobileNetV2 backbone and custom head."""
    # Load pre-trained MobileNetV2 as backbone (without top classifier, with ImageNet weights)
    base_model = keras.applications.MobileNetV2(weights='imagenet', input_shape=input_shape, include_top=False)
    base_model.trainable = False  # Freeze the backbone layers so they are not trainable:contentReference[oaicite:10]{index=10}
    # Define model architecture
    inputs = keras.Input(shape=input_shape)
    # Pass inputs through the base model. Ensure training=False so batch norm layers run in inference mode:contentReference[oaicite:11]{index=11}.
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)          # Pool the features to a vector:contentReference[oaicite:12]{index=12}
    x = layers.Dense(128, activation='relu')(x)     # Optional dense layer for learning combination of features
    x = layers.Dropout(0.3)(x)                      # Dropout for regularization
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Sigmoid output for binary classification (good vs bad)
    model = keras.Model(inputs, outputs)
    # Compile the model with binary cross-entropy loss and accuracy metric
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_ds, val_ds, category_name):
    """Trains the given model on the train_ds, using val_ds for validation. Saves best model and returns it."""
    # Set up callbacks for early stopping and model checkpointing
    callbacks = []
    # Early stopping: stop if validation loss doesn't improve for a while
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    callbacks.append(early_stop)
    # Model checkpoint: save the best model to a file
    checkpoint_path = f'output_models/best_{category_name}_model.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                                save_best_only=True, verbose=1)
    callbacks.append(checkpoint)
    # Train the model
    history = model.fit(train_ds, 
                        epochs=50, 
                        validation_data=val_ds, 
                        callbacks=callbacks,
                        verbose=1)
    # The model instance is updated with the best weights due to restore_best_weights=True
    return model, history
