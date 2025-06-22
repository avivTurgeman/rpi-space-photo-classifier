import os
import torch
from torch.utils.data import DataLoader

from utils.logger import get_logger
from data.dataset import SatelliteImageDataset
import torchvision.transforms as transforms

from models.horizon_model import HorizonModel
from models.star_model import StarModel
from models.quality_model import QualityModel

logger = get_logger(__name__)

def evaluate_model(task_name):
    """Evaluate the trained model for the given task on the test dataset."""
    model_path = os.path.join('models', f"{task_name}_model.pth")
    data_dir = os.path.join('data', task_name, 'test')
    if not os.path.isfile(model_path):
        logger.error(f"Model weights not found at {model_path}. Train the model first.")
        return
    if not os.path.isdir(data_dir):
        logger.error(f"Test data directory not found: {data_dir}")
        return

    # Load the dataset
    test_dataset = SatelliteImageDataset(data_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate the model and load weights
    if task_name == 'horizon':
        model = HorizonModel()
    elif task_name == 'star':
        model = StarModel()
    elif task_name == 'quality':
        model = QualityModel()
    else:
        logger.error(f"Unknown task: {task_name}")
        return
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # load to CPU
    model.eval()
    logger.info(f"Model loaded from {model_path}. Beginning evaluation on test set...")

    # Evaluate accuracy
    correct = 0
    total = 0
    # We don't need gradients for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # Predicted class is the index of the max logit
            _, predicted = torch.max(outputs, 1)  #:contentReference[oaicite:23]{index=23}
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0.0
    logger.info(f"Test Accuracy for task '{task_name}': {accuracy:.2f}% ({correct}/{total} correct)")
    print(f"Accuracy on {task_name} test set: {accuracy:.2f}%")
