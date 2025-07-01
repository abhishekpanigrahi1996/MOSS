import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random

# Set random seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# Define transform to convert to tensor
transform = transforms.Compose([transforms.ToTensor()])  # Scales to [0,1]

# Load CIFAR-10 training set
dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Create a dictionary to store one image per class
class_images = {i: None for i in range(10)}

# Shuffle dataset and pick one image per class
indices = list(range(len(dataset)))
random.shuffle(indices)

for idx in indices:
    image, label = dataset[idx]
    if class_images[label] is None:
        class_images[label] = image
    if all(img is not None for img in class_images.values()):
        break

# Stack and average the images
images_tensor = torch.stack([img for img in class_images.values()])
mixup_image = images_tensor.mean(dim=0)  # shape: [3, 32, 32]

# Convert to PIL Image for saving
to_pil = transforms.ToPILImage()
mixup_pil = to_pil(mixup_image)

# Save to file
mixup_pil.save("cifar10_mixup.png")
print("Saved mixup image as 'cifar10_mixup.png'")
