import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.models

from utils.evaluate_model import evaluate
from utils.general_utils import replace_specific_layer_2, replace_layer_by_name
from low_rank_layers.lr_layer_base import LowRankLayerBase
import argparse
from utils.illustration_utils import get_matrices_recursive, get_condition_number

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

parser.add_argument(
    "--layer_name",
    type=str,
    default="classifier.3",
    help="Layer to truncate",
)

parser.add_argument(
    "--layer_rank",
    type=int,
    default=3000,
    help="Number of ranks to be truncated. 0 means no truncation",
)

parser.add_argument(
    "--flatten_dim",
    type=int,
    default=0,
    choices=[0, 1, 2, 3],
    help="Dimension to flatten convolution tensors",
)


# Parse the arguments
args = parser.parse_args()

# Access the arguments
pretrained_weights = args.pretrained_weights
model_name = args.model
layer_name = args.layer_name
layer_rank = args.layer_rank

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


if layer_name != "none":
    # print("Truncating layer", layer_name)
    replace_layer_by_name(
        network_model,
        layer_name,
        LowRankLayerBase,
        max_rank=layer_rank,
    )

if pretrained_weights == "imagenet":
    avg_loss, avg_accuracy = evaluate(
        model=network_model, imagenet_path="../../ml_training_data/imagenet/tmp/"
    )
    # print("avg acc", avg_accuracy)
    # print("avg loss", avg_loss)
    matrices = get_matrices_recursive(network_model, flatten_dim=args.flatten_dim)
    print([matrix[0].shape for matrix in matrices])
    total_condition_product, condition_numbers = get_condition_number(matrices)

out_str = ""
out_str += str(layer_rank) + "," + str(avg_accuracy) + "," + str(avg_loss)
out_str += str(total_condition_product.cpu().numpy())
for condition_number in condition_numbers:
    out_str += "," + str(condition_number.cpu().numpy())
out_str += "\n"
# print(out_str)

with open("results/evaluate_model_val_accuracy_selective_lr.csv", "a") as f:
    f.write(out_str)
# print(out_str)

# print("--- Finished ---")
