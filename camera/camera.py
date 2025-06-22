#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime
from utils import metadata  # Import metadata utilities for load/save

### CONFIG – edit these to match your setup ###
# Directory to save photos and metadata
PHOTO_DIR   = "/home/admin/Desktop/picture_classifier/photos"
BURST_COUNT = 3  # default number of photos for burst mode
###############################################

# Determine which camera command is available (raspistill for legacy camera, libcamera-still for newer)
def detect_cam_tool():
    for cmd in ("raspistill", "libcamera-still"):
        if subprocess.run(["which", cmd], capture_output=True).returncode == 0:
            return cmd
    # If neither tool is found, prompt user to install one
    print("Error: neither raspistill nor libcamera-still found.")
    print("Install legacy tool: sudo apt update && sudo apt install libraspberrypi-bin")
    print("Or install libcamera apps: sudo apt update && sudo apt install libcamera-apps")
    raise SystemExit(1)

# Select the camera capture command
CAM_TOOL = detect_cam_tool()

def ensure_dirs():
    """Create photo directory and metadata file if they don't exist."""
    os.makedirs(PHOTO_DIR, exist_ok=True)
    meta_path = os.path.join(PHOTO_DIR, "metadata.json")
    if not os.path.isfile(meta_path):
        # Initialize an empty metadata file
        metadata.save_meta({})

def timestamp():
    """Return current date/time as a timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def capture(filename):
    """
    Capture a single image with the camera and save to PHOTO_DIR with the given filename.
    Returns the full path of the saved image, or None if capture failed.
    """
    path = os.path.join(PHOTO_DIR, filename)
    # Build the subprocess command for capturing an image
    cmd = [CAM_TOOL, "-o", path]
    # If using libcamera-still, add "--nopreview" to avoid preview window
    if CAM_TOOL == "libcamera-still":
        cmd.insert(1, "--nopreview")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Error capturing image:", e)
        return None
    return path

def take_single():
    """Capture a single photo and update metadata."""
    fname = f"photo_{timestamp()}.jpg"
    path = capture(fname)
    if not path:
        return  # capture failed
    # Load existing metadata and add new entry
    meta = metadata.load_meta()
    meta[fname] = {
        "datetime": datetime.now().isoformat(),
        "classification": None  # not classified yet
    }
    metadata.save_meta(meta)
    print(f"Saved {fname}")

def take_burst():
    """Capture a burst of BURST_COUNT photos in succession and update metadata for each."""
    meta = metadata.load_meta()
    for i in range(1, BURST_COUNT + 1):
        fname = f"burst_{timestamp()}_{i}.jpg"
        path = capture(fname)
        if path:
            meta[fname] = {
                "datetime": datetime.now().isoformat(),
                "classification": None
            }
            print(f"→ {fname}")
    metadata.save_meta(meta)
