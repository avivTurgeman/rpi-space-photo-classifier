import os
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as transforms

class SatelliteImageDataset(Dataset):
    """Custom Dataset for satellite image classification tasks.
       Expects directory structure: root/class_a, root/class_b, ... with images inside."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # List all image file paths and their labels
        self.file_paths = []
        self.labels = []
        self.class_names = []
        # Determine class subfolders
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_names.append(class_name)
                class_index = len(self.class_names) - 1
                # Iterate over files in the class directory
                for fname in os.listdir(class_path):
                    fpath = os.path.join(class_path, fname)
                    if os.path.isfile(fpath):
                        # (Optionally, filter by image extension if needed)
                        self.file_paths.append(fpath)
                        self.labels.append(class_index)
        # If no transform provided, define a default (no augmentation, just tensor conversion)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # normalize to [-1,1]:contentReference[oaicite:13]{index=13}
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load image and apply transformations
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
