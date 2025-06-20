#!/usr/bin/env python3
import os, sys, json, subprocess
from datetime import datetime

### CONFIG — edit these to match your setup ###
PHOTO_DIR        = "/home/admin/Desktop/picture_classifier/photos"
METADATA_FILE    = os.path.join(PHOTO_DIR, "metadata.json")
BURST_COUNT      = 3
MAC_USER         = "isiah"
MAC_DOWNLOAD_DIR = "/Users/isiah/Downloads"
###############################################

# detect which capture command to use
def detect_cam_tool():
    for cmd in ("raspistill", "libcamera-still"):
        if subprocess.run(["which", cmd], capture_output=True).returncode == 0:
            return cmd
    print("Error: neither raspistill nor libcamera-still found.")
    print("Install legacy tool: sudo apt update && sudo apt install libraspberrypi-bin")
    print("Or install libcamera apps: sudo apt update && sudo apt install libcamera-apps")
    sys.exit(1)

CAM_TOOL = detect_cam_tool()

def ensure_dirs():
    os.makedirs(PHOTO_DIR, exist_ok=True)
    if not os.path.isfile(METADATA_FILE):
        with open(METADATA_FILE, "w") as f:
            json.dump({}, f)

def load_meta():
    with open(METADATA_FILE) as f:
        return json.load(f)

def save_meta(meta):
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def capture(filename):
    path = os.path.join(PHOTO_DIR, filename)
    cmd = [CAM_TOOL, "-o", path]
    # for libcamera-still you might want to add "--nopreview"
    if CAM_TOOL == "libcamera-still":
        cmd.insert(1, "--nopreview")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Error capturing:", e)
        return None
    return path

def take_single():
    fname = f"photo_{timestamp()}.jpg"
    path = capture(fname)
    if not path: return
    meta = load_meta()
    meta[fname] = {"datetime": datetime.now().isoformat(), "classification": None}
    save_meta(meta)
    print("Saved", fname)

def take_burst():
    meta = load_meta()
    for i in range(1, BURST_COUNT+1):
        fname = f"burst_{timestamp()}_{i}.jpg"
        path = capture(fname)
        if path:
            meta[fname] = {"datetime": datetime.now().isoformat(), "classification": None}
            print("→", fname)
    save_meta(meta)

def classify_image(path):
    """ TODO: hook in your real model here """
    return None

def run_classification():
    meta = load_meta()
    updated = False
    for fname, info in meta.items():
        if info["classification"] is None:
            p = os.path.join(PHOTO_DIR, fname)
            try:
                label = classify_image(p)
                meta[fname]["classification"] = label or "Unknown"
                print(f"{fname}: {meta[fname]['classification']}")
            except Exception as e:
                print("Failed:", fname, e)
            updated = True
    if updated: save_meta(meta)

def list_photos():
    meta = load_meta()
    if not meta:
        print("No photos yet.")
        return
    print(f"{'FILE':30}  {'DATE/TIME':20}  CLASS")
    print("-"*70)
    for fname, info in sorted(meta.items(), key=lambda x: x[1]["datetime"]):
        dt = info["datetime"]
        cl = info["classification"] or "Not classified"
        print(f"{fname:30}  {dt:20}  {cl}")

def view_latest():
    meta = load_meta()
    if not meta:
        print("No photos to view.")
        return
    latest = max(meta.items(), key=lambda x: x[1]["datetime"])[0]
    src = os.path.join(PHOTO_DIR, latest)
    ssh_client = os.environ.get("SSH_CLIENT", "")
    client_ip = ssh_client.split()[0] if ssh_client else None
    if not client_ip:
        print("Cannot detect client IP; ensure SSH_CLIENT is set.")
        return
    dest = f"{MAC_USER}@{client_ip}:{MAC_DOWNLOAD_DIR}"
    try:
        subprocess.run(["scp", src, dest], check=True)
        print(f"Transferred {latest} to {dest}")
    except subprocess.CalledProcessError as e:
        print("SCP failed:", e)

def menu():
    opts = {
        "1": ("Take single photo", take_single),
        "2": (f"Take burst ({BURST_COUNT})", take_burst),
        "3": ("Classify", run_classification),
        "4": ("List photos", list_photos),
        "5": ("View latest on Mac", view_latest),
        "6": ("Exit", lambda: sys.exit(0)),
    }
    while True:
        print("\n=== MENU ===")
        for k,(d,_) in opts.items():
            print(f"{k}) {d}")
        choice = input("Select: ").strip()
        if choice in opts:
            opts[choice][1]()
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    ensure_dirs()
    menu()
