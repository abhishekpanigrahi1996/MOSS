import torch
import torch.nn as nn
import torch.optim as optim

from low_rank_layers.layer_utils import replace_layer_by_name
import argparse
from low_rank_training.trainer import Trainer
from utils.data_utils import choose_dataset
from utils.general_utils import get_available_device


def main():
    args = argument_parser()
    print_user_choices(args)
    device = get_available_device(multi_gpu=args.multi_gpu)

    # Access the arguments
    network_model, output_dir = get_pretrained_model(
        model_name=args.model, pretrained_weights=args.pretrained_weights
    )
    lr_layers = []

    if args.layer_names:  # Check if there are layers to replace
        # print("Truncating layer", layer_name)
        for layer_name in args.layer_names:
            lr_layers.append(
                replace_layer_by_name(
                    network_model,
                    layer_name,
                    max_rank=args.max_rank,
                    init_rank=args.initial_rank,
                    tol=args.tol,
                )
            )

    print("Replaced layer", lr_layers)
    # print(network_model)

    train_loader, val_loader, _ = get_dataset(
        args.pretrained_weights, args.batch_size, args.imagenet_path
    )
    optimizer = optim.SGD(network_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=5e-4
    )

    t = Trainer(
        model=network_model,
        train_loader=train_loader,
        test_loader=val_loader,
        scheduler=scheduler,
        optimizer=optimizer,
        device=device,
        lr_layers=lr_layers,
        args=args,
    )
    t.train()


def argument_parser():
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
        "--imagenet_path",
        type=str,
        default="../../ml_training_data/imagenet",
        choices=["imagenet", "ucm", "aid"],
        help="Choose model pretrained weights",
    )

    # ---- Low Rank Training Hyperparameters ----

    parser.add_argument(
        "--layer_names",
        type=str,
        nargs="+",  # This tells argparse to expect multiple arguments for this flag
        default=["classifier.3", "classifier.0"],
        help="Layers for DLRT",
    )

    parser.add_argument(
        "--initial_rank",
        type=int,
        default=200,
        help="Initial rank of the low-rank layers",
    )
    parser.add_argument(
        "--max_rank",
        type=int,
        default=400,
        help="Maximum rank of the low-rank layers",
    )

    parser.add_argument(
        "--tol", type=float, default=0.05, help="Tolerance for low-rank truncation step"
    )

    parser.add_argument(
        "--num_local_iter",
        type=int,
        default=5,
        help="Number of coefficient updates between basis updates",
    )

    # --- Basic training hyperparameters ---
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for training"
    )

    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="Wegith decay for training"
    )

    parser.add_argument(
        "--wandb", type=bool, default=0, help="Activate wandb logging: 0=no, 1=yes"
    )

    parser.add_argument(
        "--multi_gpu", type=bool, default=0, help="Activate multi gpu: 0=no, 1=yes"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


def get_pretrained_model(model_name, pretrained_weights):
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

            network_model = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
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

            network_model = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1
            )
            output_dir = "resnet34_Imagenet1K_V1"
        else:
            exit("Weights aid not implemented")

    return network_model, output_dir


def get_dataset(pretrained_weights, batch_size, imagenet_path):
    if pretrained_weights == "imagenet":
        train_loader, val_loader, test_loader = choose_dataset(
            dataset_choice=4,  # imagenet
            train_batch_size=batch_size,
            datapath=imagenet_path,  # Change this to your imagenet path
            num_workers=1,
        )
    elif pretrained_weights == "ucm":
        train_loader, val_loader, test_loader = choose_dataset(
            dataset_choice=5,  # ucm
            train_batch_size=batch_size,
            datapath="",  # Uses Huggingface datasets by default
            num_workers=1,
        )
    else:
        exit("Weights aid not implemented")

    return train_loader, val_loader, test_loader


def print_user_choices(args):
    print("\n--- Model Selection ---")
    print(f"Model: {args.model}")
    print(f"Pretrained Weights: {args.pretrained_weights}")

    print("\n--- Low Rank Training Hyperparameters ---")
    print(f"Layer Names: {args.layer_names}")
    print(f"Initial Rank: {args.initial_rank}")
    print(f"Max Rank: {args.max_rank}")
    print(f"Tolerance: {args.tol}")
    print(f"Number of Local Iterations: {args.num_local_iter}")

    print("\n--- Basic Training Hyperparameters ---")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Momentum: {args.momentum}")
    print(f"Weight Decay: {args.weight_decay}")

    print("\n--- Logging ---")
    print(f"WandB Logging: {'Enabled' if args.wandb else 'Disabled'}")


if __name__ == "__main__":
    main()
