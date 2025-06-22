import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from utils.config import config
from utils.logger import get_logger
from data.dataset import SatelliteImageDataset
import torchvision.transforms as transforms

# Import model classes
from models.horizon_model import HorizonModel
from models.star_model import StarModel
from models.quality_model import QualityModel

logger = get_logger(__name__)

def train_model(task_name):
    """Train the model for the specified task ('horizon', 'star', or 'quality')."""
    # Configuration (hyperparameters)
    lr = config.get('learning_rate', 0.001)
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 10)
    data_dir = os.path.join('data', task_name)

    # Prepare dataset and data loader
    train_dir = os.path.join(data_dir, 'train')
    # Optionally, you could add code to split train into train/val if no separate val set.
    # For simplicity, assume a 'train' folder exists (and optionally 'val').
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_dataset = SatelliteImageDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # If validation directory exists, prepare val loader
    val_loader = None
    val_dir = os.path.join(data_dir, 'val')
    if os.path.isdir(val_dir):
        val_dataset = SatelliteImageDataset(val_dir, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Instantiate the model for this task
    if task_name == 'horizon':
        model = HorizonModel()
    elif task_name == 'star':
        model = StarModel()
    elif task_name == 'quality':
        model = QualityModel()
    else:
        logger.error(f"Unknown task: {task_name}")
        return

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # using SGD with momentum:contentReference[oaicite:16]{index=16}

    logger.info(f"Starting training for task '{task_name}' for {epochs} epochs...")
    model.train()  # set model to training mode
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backpropagate and update weights
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Print/log every 100 mini-batches
            if (i+1) % 100 == 0:
                avg_loss = running_loss / 100
                logger.info(f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Optionally, evaluate on validation set each epoch
        if val_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)  # get index of max logit
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = 100 * correct / total if total > 0 else 0
            logger.info(f"Epoch {epoch}: Validation Accuracy = {acc:.2f}%")
            model.train()  # back to train mode

    logger.info("Training completed. Saving model...")
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', f"{task_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
