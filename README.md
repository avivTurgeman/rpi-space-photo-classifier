# rpi-space-photo-classifier
Lightweight CLI for Raspberry Pi Zero 2 to capture single/burst photos, index metadata, run image-classification stubs, and seamlessly transfer the latest shot to a ground desktop over SSH.
# rpi-photo-cli

Lightweight CLI for Raspberry Pi Zero 2 to capture photos, index metadata, run classification, and transfer to desktop (specifically this one is for Mac) over SSH.

---

## Features

* **Single Photo Capture**: Grab a photo with one command.
* **Burst Mode**: Capture multiple shots (configurable count).
* **Metadata Indexing**: Maintains `metadata.json` with timestamp and classification status.
* **Classification Stub**: Integrate your own model via the `classify_image()` stub.
* **Photo Listing**: View all photos with date/time and classification.
* **Seamless Transfer**: Automatically SCP the latest photo to your Mac.

---

## Requirements

* Raspberry Pi OS (Lite or Desktop)
* Python 3
* `raspistill` (legacy) or `libcamera-still`
* SSH access from your desktop

---


## Configuration

At the top of `classifier.py`, adjust these constants:

```python
PHOTO_DIR        = "/home/admin/Desktop/picture_classifier/photos"
BURST_COUNT      = 3
MAC_USER         = "isiah"
MAC_DOWNLOAD_DIR = "/Users/isiah/Downloads"
```

---

## Usage

Run the CLI over SSH:

```bash
./classifier.py
```

**Menu Options**:

1. Take single photo
2. Take burst (3 shots)
3. Classify unclassified photos
4. List photos
5. View latest on Mac
6. Exit

---

## Pictures


