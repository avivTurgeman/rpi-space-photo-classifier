import os
import json

# Path configuration for photos and metadata
PHOTO_DIR = "/home/admin/Desktop/picture_classifier/photos"
METADATA_FILE = os.path.join(PHOTO_DIR, "metadata.json")

def load_meta():
    """Load the metadata JSON file and return it as a dictionary. If file is missing, return empty dict."""
    try:
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # If metadata file doesn't exist, return empty metadata
        return {}

def save_meta(meta):
    """Save the metadata dictionary to the JSON file (pretty-printed)."""
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def update_meta(filename, info):
    """
    Update the metadata for a single image. 
    `info` should be a dict containing keys like "datetime" or "classification".
    """
    meta = load_meta()
    meta[filename] = info
    save_meta(meta)
