import torch
import torch.nn as nn
import torch.nn.functional as F

class StarModel(nn.Module):
    """CNN for Star Detection (binary classification: stars vs no stars)."""
    def __init__(self):
        super(StarModel, self).__init__()
        # Using a similar architecture to HorizonModel
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        flat_features = 16 * 13 * 13  # assuming 64x64 input as before
        self.fc1 = nn.Linear(flat_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2 classes: [No Stars, Stars]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
