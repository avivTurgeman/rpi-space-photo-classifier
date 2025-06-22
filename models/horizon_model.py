import torch
import torch.nn as nn
import torch.nn.functional as F

class HorizonModel(nn.Module):
    """CNN for Horizon Detection (binary classification: horizon vs no horizon)."""
    def __init__(self):
        super(HorizonModel, self).__init__()
        # Convolutional layers: small kernel (5x5) with few filters
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)     # 3 input channels (RGB), 6 filters
        self.pool = nn.MaxPool2d(2, 2)                  # 2x2 pooling
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # 16 filters
        # Compute the flattened feature size after conv layers:
        # Assuming input images ~64x64 pixels for calculation (adjust if different).
        # After conv1 (5x5, no padding) -> ~60x60, pool -> 30x30
        # After conv2 (5x5) -> ~26x26, pool -> 13x13
        flat_features = 16 * 13 * 13
        # Fully connected layers
        self.fc1 = nn.Linear(flat_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2 output classes: [No Horizon, Horizon]

    def forward(self, x):
        # Two conv/pool layers with ReLU activation
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor (except batch dimension)
        x = torch.flatten(x, 1)
        # Two fully-connected layers with ReLU, then output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # raw scores for 2 classes
        return x
