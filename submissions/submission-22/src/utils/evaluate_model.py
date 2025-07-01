from utils.data_utils import choose_dataset
from utils.general_utils import get_available_device

import torch
import torch.nn.functional as F


def evaluate(model, train_loader, val_loader, test_loader):
    device = get_available_device()

    model.eval()  # Set the model to evaluation mode
    model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Disable gradient computation to save memory and improve speed
    with torch.no_grad():
        for batch in val_loader:
            # Assuming each batch is a tuple (inputs, labels)
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)  # Or use the criterion defined

            # Accumulate loss
            total_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += inputs.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy
