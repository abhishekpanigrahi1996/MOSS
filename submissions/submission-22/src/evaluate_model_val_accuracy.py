import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.models

from utils.evaluate_model import evaluate
from utils.data_utils import choose_dataset

import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Set model parameters")
parser.add_argument(
    "--model",
    type=str,
    default="VGG16",
    choices=["VGG16", "ResNet18", "ResNet34"],
    help="Model to use",
)

parser.add_argument(
    "--pretrained_weights",
    type=str,
    default="imagenet",
    choices=["imagenet", "ucm", "aid"],
    help="Choose model pretrained weights",
)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
pretrained_weights = args.pretrained_weights
model_name = args.model


if model_name == "VGG16":
    if pretrained_weights == "ucm":
        from tools import model as models

        network_model = models.vgg16(pretrained=False)
        ucm_classes = 21
        num_classes = ucm_classes
        network_model.classifier._modules["6"] = nn.Linear(4096, num_classes)

        PRE_TRAIN_MODEL_PATH = "models/UCM/Pretrain/vgg16/epoch_10_OA_9200.pth"

        saved_state_dict = torch.load(PRE_TRAIN_MODEL_PATH)
        network_model.load_state_dict(saved_state_dict, strict=False)
        output_dir = "vgg16_UCM"
    elif pretrained_weights == "imagenet":
        import torchvision.models as models

        network_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        output_dir = "vgg16_Imagenet1K_V1"
    else:
        exit("Weights aid not implemented")

elif model_name == "ResNet18":
    if pretrained_weights == "ucm":
        from tools import model as models

        network_model = models.resnet18(pretrained=False)
        ucm_classes = 21
        num_classes = ucm_classes
        network_model.fc = nn.Linear(512, num_classes)

        PRE_TRAIN_MODEL_PATH = "models/UCM/Pretrain/resnet18/epoch_10_OA_9666.pth"

        saved_state_dict = torch.load(PRE_TRAIN_MODEL_PATH)
        network_model.load_state_dict(saved_state_dict, strict=False)
        output_dir = "resnet18_UCM"
    elif pretrained_weights == "imagenet":
        import torchvision.models as models

        network_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        output_dir = "resnet18_Imagenet1K_V1"
    else:
        exit("Weights aid not implemented")
elif model_name == "ResNet34":
    if pretrained_weights == "ucm":
        from tools import model as models

        network_model = models.resnet34(pretrained=False)
        ucm_classes = 21
        num_classes = ucm_classes
        network_model.fc = nn.Linear(512, num_classes)

        PRE_TRAIN_MODEL_PATH = "models/UCM/Pretrain/resnet34/epoch_10_OA_9657.pth"

        saved_state_dict = torch.load(PRE_TRAIN_MODEL_PATH)
        network_model.load_state_dict(saved_state_dict, strict=False)
        output_dir = "resnet34_UCM"
    elif pretrained_weights == "imagenet":
        import torchvision.models as models

        network_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        output_dir = "resnet34_Imagenet1K_V1"
    else:
        exit("Weights aid not implemented")

# print(network_model)
# print(model_name)
# print(pretrained_weights)

if pretrained_weights == "imagenet":
    train_loader, val_loader, test_loader = choose_dataset(
        dataset_choice=4,  # imagenet
        train_batch_size=512,
        datapath="../../ml_datasets/imagenet/",  # Change this to your imagenet path
        num_workers=1,
    )
elif pretrained_weights == "ucm":
    train_loader, val_loader, test_loader = choose_dataset(
        dataset_choice=5,  # ucm
        train_batch_size=512,
        datapath="",  # Uses Huggingface datasets by default
        num_workers=1,
    )
else:
    exit("Weights aid not implemented")

avg_loss, avg_accuracy = evaluate(
    model=network_model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

print("avg acc", avg_accuracy)
print("avg loss", avg_loss)
