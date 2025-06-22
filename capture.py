#!/usr/bin/env python3
import os
import sys
import subprocess

from camera import camera  # import camera module functions
from utils import metadata  # import metadata utilities
from inference.predict import classify_image

### CONFIG – edit these to match your setup ###
MAC_USER         = "isiah"  # Your Mac username for SCP transfer
MAC_DOWNLOAD_DIR = "/Users/isiah/Downloads"  # Directory on Mac to receive the photo
###############################################

def run_classification():
    """Run classification on all unclassified images and update metadata with results."""
    meta = metadata.load_meta()
    updated = False
    for fname, info in meta.items():
        if info.get("classification") is None:
            image_path = os.path.join(metadata.PHOTO_DIR, fname)
            try:
                label = classify_image(image_path)
                # Save the classification label (or "Unknown" if classify_image returned falsy)
                meta[fname]["classification"] = label or "Unknown"
                print(f"{fname}: {meta[fname]['classification']}")
            except Exception as e:
                print(f"Failed: {fname} - {e}")
            else:
                updated = True
    if updated:
        metadata.save_meta(meta)

def list_photos():
    """Print a list of all photos with their timestamp and classification."""
    meta = metadata.load_meta()
    if not meta:
        print("No photos yet.")
        return
    print(f"{'FILE':30}  {'DATE/TIME':20}  CLASSIFICATION")
    print("-" * 70)
    # Sort by datetime field
    for fname, info in sorted(meta.items(), key=lambda x: x[1]["datetime"]):
        dt = info.get("datetime", "")
        cls = info.get("classification") or "Not classified"
        print(f"{fname:30}  {dt:20}  {cls}")

def view_latest():
    """Transfer the latest photo to the Mac using SCP (requires SSH connection from Mac)."""
    meta = metadata.load_meta()
    if not meta:
        print("No photos to view.")
        return
    # Find the latest photo by datetime
    latest_file, latest_info = max(meta.items(), key=lambda x: x[1]["datetime"])
    src_path = os.path.join(metadata.PHOTO_DIR, latest_file)
    # Get the SSH client IP from environment (set when connected via SSH)
    ssh_client = os.environ.get("SSH_CLIENT", "")
    client_ip = ssh_client.split()[0] if ssh_client else None
    if not client_ip:
        print("Cannot detect client IP. Make sure you're SSHed in from the Mac (SSH_CLIENT env not set).")
        return
    dest = f"{MAC_USER}@{client_ip}:{MAC_DOWNLOAD_DIR}"
    try:
        subprocess.run(["scp", src_path, dest], check=True)
        print(f"Transferred {latest_file} to {dest}")
    except subprocess.CalledProcessError as e:
        print(f"SCP transfer failed: {e}")

def main_menu():
    """Display the menu and handle user choices."""
    # Ensure directories and metadata file exist before starting
    camera.ensure_dirs()
    options = {
        "1": ("Take single photo", camera.take_single),
        "2": (f"Take burst ({camera.BURST_COUNT}) photos", camera.take_burst),
        "3": ("Run classification on unclassified images", run_classification),
        "4": ("List all photos", list_photos),
        "5": ("View latest photo on Mac (transfer via SCP)", view_latest),
        "6": ("Exit", lambda: sys.exit(0))
    }
    while True:
        print("\n=== MENU ===")
        for key, (description, _) in options.items():
            print(f"{key}) {description}")
        choice = input("Select an option: ").strip()
        if choice in options:
            # Execute the function associated with the choice
            options[choice][1]()
        else:
            print("Invalid choice, please try again.")