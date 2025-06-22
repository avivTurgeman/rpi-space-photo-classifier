import torch
import torch.nn as nn
import torch.nn.functional as F

class QualityModel(nn.Module):
    """CNN for Image Quality Classification (binary: good vs poor quality)."""
    def __init__(self):
        super(QualityModel, self).__init__()
        # We can use the same structure, possibly with a smaller first layer
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        flat_features = 16 * 13 * 13
        self.fc1 = nn.Linear(flat_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2 classes: [Poor, Good] quality

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
