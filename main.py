import os
import sys
import cv2
import argparse
import configparser
from predict import predict_stars, predict_horizon

# === Configuration Setup ===
config_file = "config.conf"
config = configparser.ConfigParser()
if os.path.exists(config_file):
    config.read(config_file)
else:
    # If config file doesn't exist, create it with default values
    config['global'] = {'satlla_id': '0', 'bootcount': '0', 'resetlogfactor': '0', 'serialpath': '/dev/serial0'}
    config['mission'] = {'missioncount': '0', 'piccount': '0'}
    config['communication'] = {'lora24packets': '0'}
    config['RWCS'] = {'gpio_fet_pin': '24'}
    # Write initial config to file
    with open(config_file, 'w') as f:
        config.write(f)

# Increment the boot count on every startup
boot_count = int(config['global']['bootcount'])
boot_count += 1
config['global']['bootcount'] = str(boot_count)

# === Command-line Argument Parsing ===
parser = argparse.ArgumentParser(description="Satellite OBC Main Program")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-c', '--capture', action='store_true', help="Capture an image and classify it (Capture and Classify mission)")
group.add_argument('-s', '--send', action='store_true', help="Compress all saved images and send (Compress and Send mission)")
args = parser.parse_args()

# Ensure image directories exist
TO_SEND_DIR = "images_to_send"
STARS_DIR = os.path.join(TO_SEND_DIR, "stars")
HORIZON_DIR = os.path.join(TO_SEND_DIR, "horizon")
os.makedirs(STARS_DIR, exist_ok=True)
os.makedirs(HORIZON_DIR, exist_ok=True)

# === Mission 1: Capture and Classify ===
if args.capture:
    # Increment mission count (each capture attempt is a mission run)
    mission_count = int(config['mission']['missioncount'])
    mission_count += 1
    config['mission']['missioncount'] = str(mission_count)
    print(f"[Mission] Capture and Classify (mission #{mission_count}) starting...")

    # Initialize camera capture using OpenCV
    cap = cv2.VideoCapture(0)  # Open default camera (Pi Camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[Error] Camera capture failed!")
    else:
        # Check if the captured image is too blurry
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Using Laplacian variance method: low variance indicates blur:contentReference[oaicite:4]{index=4}
        blur_threshold = 100.0
        if laplacian_var < blur_threshold:
            print(f"[Info] Image is blurry (Laplacian var={laplacian_var:.2f} < {blur_threshold}), deleting image.")
            # We do not save blurry images
        else:
            print(f"[Info] Image is sharp enough (Laplacian var={laplacian_var:.2f}), classifying...")
            # Determine classification using the models
            is_star = predict_stars(frame)   # True if image is of stars
            is_horizon = False
            if not is_star:
                is_horizon = predict_horizon(frame)  # True if image is of Earth's horizon

            if is_star:
                classification = "stars"
            elif is_horizon:
                classification = "horizon"
            else:
                classification = None

            if classification is None:
                print("[Info] Image did not match stars or horizon categories. Discarding image.")
                # (Not saving the image since it's neither stars nor horizon)
            else:
                # Save image to the corresponding folder
                pic_count = int(config['mission']['piccount'])
                pic_count += 1
                filename = f"image_{pic_count}.jpg"
                save_path = os.path.join(STARS_DIR if classification == "stars" else HORIZON_DIR, filename)
                cv2.imwrite(save_path, frame)  # save the captured frame as JPEG file
                config['mission']['piccount'] = str(pic_count)
                print(f"[Info] Image classified as '{classification}'. Saved to {save_path}.")
                
# === Mission 2: Compress and Send ===
elif args.send:
    print("[Mission] Compress and Send starting...")
    # Collect all image files from the stars and horizon folders
    image_files = []
    for fname in os.listdir(STARS_DIR):
        image_files.append(os.path.join(STARS_DIR, fname))
    for fname in os.listdir(HORIZON_DIR):
        image_files.append(os.path.join(HORIZON_DIR, fname))
    if not image_files:
        print("[Info] No images to send. Exiting.")
    else:
        # Read and combine all images into one bytearray
        combined_bytes = bytearray()
        for filepath in image_files:
            with open(filepath, 'rb') as f:
                data = f.read()
                combined_bytes.extend(data)
        # Simulate sending the data to Arduino for broadcast.
        # In a real satellite, we'd transmit via serial/LoRa; here we just print it out.
        print(f"[Data] Combined image bytearray ({len(combined_bytes)} bytes):")
        print(combined_bytes)  # Output the raw bytearray as a sign of transmission
        # Clear the sent images from storage
        for filepath in image_files:
            os.remove(filepath)
        print(f"[Info] Sent {len(image_files)} images and cleared the folders.")

# Finally, save updated config values back to file
with open(config_file, 'w') as f:
    config.write(f)
print("[Info] Updated config.conf saved. BootCount =", config['global']['bootcount'],
      "MissionCount =", config['mission']['missioncount'],
      "PicCount =", config['mission']['piccount'])
