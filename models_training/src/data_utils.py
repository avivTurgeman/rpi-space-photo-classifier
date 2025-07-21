import os
import random
import tensorflow as tf

# Set seeds for reproducibility across runs
random.seed(42)
tf.random.set_seed(42)

# Constants for image dimensions and batch processing
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

def load_and_preprocess_image(path):
    # Read image file from disk
    image_data = tf.io.read_file(path)
    # Decode image bytes into a float32 tensor with 3 color channels
    image = tf.io.decode_image(image_data, channels=3, dtype=tf.dtypes.float32)
    # Explicitly set shape so subsequent ops know dimensions
    image.set_shape([None, None, 3])
    # Resize image to fixed size for model input
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # Normalize pixel values from [0,255] to [-1,1], matching MobileNetV2 requirements
    image = (image / 127.5) - 1.0
    return image

def get_datasets_for_category(data_dir, category):
    # Build paths for 'good' and 'bad' subdirectories
    category_path = os.path.join(data_dir, category)
    good_dir = os.path.join(category_path, "good")
    bad_dir = os.path.join(category_path, "bad")
    prefix = 'S' if category == 'stars' else 'H' if category == 'horizon' else ''

    # List all image file paths
    good_images = [os.path.join(good_dir, f) for f in os.listdir(good_dir)]
    bad_images = [os.path.join(bad_dir, f) for f in os.listdir(bad_dir)]

    # Shuffle lists to randomize split order
    random.shuffle(good_images)
    random.shuffle(bad_images)

    # Count samples in each class
    num_good = len(good_images)
    num_bad = len(bad_images)

    # Determine split sizes (at least 1 per split)
    train_count_good = max(int(0.15 * num_good), 1)
    test_count_good = max(int(0.15 * num_good), 1)
    val_count_good = num_good - train_count_good - test_count_good

    train_count_bad = max(int(0.15 * num_bad), 1)
    test_count_bad = max(int(0.15 * num_bad), 1)
    val_count_bad = num_bad - train_count_bad - test_count_bad

    # Slice file lists for each subset
    train_files = good_images[:train_count_good] + bad_images[:train_count_bad]
    test_files = (
        good_images[train_count_good:train_count_good+test_count_good]
        + bad_images[train_count_bad:train_count_bad+test_count_bad]
    )
    val_files = (
        good_images[train_count_good+test_count_good:]
        + bad_images[train_count_bad+test_count_bad:]
    )

    # Create corresponding label lists (1 for good, 0 for bad)
    train_labels = [1] * train_count_good + [0] * train_count_bad
    test_labels = [1] * test_count_good + [0] * test_count_bad
    val_labels = [1] * val_count_good + [0] * val_count_bad

    # Combine and shuffle to mix classes
    combined_train = list(zip(train_files, train_labels))
    combined_test = list(zip(test_files, test_labels))
    combined_val = list(zip(val_files, val_labels))
    random.shuffle(combined_train)
    random.shuffle(combined_test)
    random.shuffle(combined_val)
    train_files, train_labels = zip(*combined_train)
    test_files, test_labels = zip(*combined_test)
    val_files, val_labels = zip(*combined_val)

    # Create TF Dataset objects from file paths and labels
    train_ds = tf.data.Dataset.from_tensor_slices((list(train_files), list(train_labels)))
    val_ds = tf.data.Dataset.from_tensor_slices((list(val_files), list(val_labels)))
    test_ds = tf.data.Dataset.from_tensor_slices((list(test_files), list(test_labels)))

    # Map preprocessing function to load and normalize images
    train_ds = train_ds.map(
        lambda path, label: (load_and_preprocess_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda path, label: (load_and_preprocess_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.map(
        lambda path, label: (load_and_preprocess_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Create a single RandomRotation layer instance to avoid variable recreation
    rotation_layer = tf.keras.layers.RandomRotation(factor=0.085)

    def augment(image, label):
        # Random horizontal flip for both categories
        image = tf.image.random_flip_left_right(image)
        # Additional vertical flip only for star images (orientation irrelevant)
        if category == "stars":
            image = tf.image.random_flip_up_down(image)
        # Apply random rotation via the shared layer
        image = rotation_layer(image, training=True)
        # Randomly adjust brightness
        image = tf.image.random_brightness(image, max_delta=0.2)
        return image, label

    # Apply augmentation only to training dataset
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    # Shuffle, batch, and prefetch for performance
    train_ds = train_ds.shuffle(buffer_size=max(len(train_files), 1))
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds