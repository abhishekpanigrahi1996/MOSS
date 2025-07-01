import os
import numpy as np
import argparse
from src.tools.utils import *
from torch import nn
from dataset.scene_dataset import *
from torch.utils import data
import src.tools.model as models
from src.low_rank_layers.layer_utils import transform_to_low_rank
from src.utils.data_utils import choose_dataset


def main(args):
    if args.dataID == 1:
        DataName = "UCM"
        num_classes = 21
        classname = (
            "agricultural",
            "airplane",
            "baseballdiamond",
            "beach",
            "buildings",
            "chaparral",
            "denseresidential",
            "forest",
            "freeway",
            "golfcourse",
            "harbor",
            "intersection",
            "mediumresidential",
            "mobilehomepark",
            "overpass",
            "parkinglot",
            "river",
            "runway",
            "sparseresidential",
            "storagetanks",
            "tenniscourt",
        )

    elif args.dataID == 2:
        DataName = "AID"
        num_classes = 30
        classname = (
            "airport",
            "bareland",
            "baseballfield",
            "beach",
            "bridge",
            "center",
            "church",
            "commercial",
            "denseresidential",
            "desert",
            "farmland",
            "forest",
            "industrial",
            "meadow",
            "mediumresidential",
            "mountain",
            "parking",
            "park",
            "playground",
            "pond",
            "port",
            "railwaystation",
            "resort",
            "river",
            "school",
            "sparseresidential",
            "square",
            "stadium",
            "storagetanks",
            "viaduct",
        )

    elif args.dataID == 3:
        DataName = "Cifar10"
        num_classes = 10
        classname = (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
    else:
        KeyError("dataset not implemented")

    adv_root_dir = (
        args.root_dir
        + DataName
        + "_adv/"
        + args.attack_func
        + "/low_rank/"
        + args.surrogate_network
        + "/"
    )

    print("Loading data...")
    print(adv_root_dir)
    print(
        "./dataset/" + DataName + "_test.txt",
    )
    print("----")

    if args.dataID in [1, 2]:
        composed_transforms = transforms.Compose(
            [
                transforms.Resize(size=(args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        adv_loader = data.DataLoader(
            scene_dataset(
                root_dir=adv_root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=composed_transforms,
                mode="adv",
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        clean_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=composed_transforms,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.dataID == 3:
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        import torchvision.transforms as T

        valid_transform = T.Compose(
            [
                transforms.Resize(size=(args.crop_size, args.crop_size)),
                T.ToTensor(),
                T.Normalize(*stats),
            ]
        )
        adv_loader = data.DataLoader(
            scene_dataset(
                root_dir=adv_root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=valid_transform,
                mode="adv",
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        clean_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=valid_transform,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    ###################Target network Definition###################
    if args.target_network == "alexnet":
        network = models.alexnet(pretrained=False)
        network.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif args.target_network == "vgg11":
        network = models.vgg11(pretrained=False)
        network.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif args.target_network == "vgg16":
        network = models.vgg16(pretrained=False)
        network.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif args.target_network == "vgg19":
        network = models.vgg19(pretrained=False)
        network.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif args.target_network == "resnet18":
        network = models.resnet18(pretrained=False)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "resnet34":
        network = models.resnet34(pretrained=True)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "resnet50":
        network = models.resnet50(pretrained=False)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "resnet101":
        network = models.resnet101(pretrained=False)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "resnext50_32x4d":
        network = models.resnext50_32x4d(pretrained=False)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "resnext101_32x8d":
        network = models.resnext101_32x8d(pretrained=False)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "densenet121":
        network = models.densenet121(pretrained=False)
        network.classifier = nn.Linear(1024, num_classes)
    elif args.target_network == "densenet169":
        network = models.densenet169(pretrained=False)
        network.classifier = nn.Linear(1664, num_classes)
    elif args.target_network == "densenet201":
        network = models.densenet201(pretrained=False)
        network.classifier = nn.Linear(1920, num_classes)
    elif args.target_network == "inception":
        network = models.inception_v3(pretrained=True, aux_logits=False)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "regnet_x_400mf":
        network = models.regnet_x_400mf(pretrained=False)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "regnet_x_8gf":
        network = models.regnet_x_8gf(pretrained=False)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "regnet_x_16gf":
        network = models.regnet_x_16gf(pretrained=False)
        network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
    elif args.target_network == "vit32b":
        from src.tools.model_vit import vit_b32
        network = vit_b32(num_classes)
        # Transform model to low-rank factorization

    network = torch.nn.DataParallel(network).cuda()

    network, lr_layers = transform_to_low_rank(
        network, max_rank=args.rmax, init_rank=10, tol=args.tol
    )

    dirpath = args.save_path_prefix + DataName + "/Pretrain/low_rank/" + args.target_network + "/"

    model_name = (
        "beta_"
        + str(args.robusteness_regularization_beta)
        + "_tol_"
        + str(args.tol)
        + "_rmax_"
        + str(args.rmax)
        + "_init_rank_"
        + str(args.init_r)
        + ".pth"
    )
    model_path_resume = os.path.join(dirpath, model_name)
    saved_state_dict = torch.load(model_path_resume)

    new_params = network.state_dict().copy()

    for i, j in zip(saved_state_dict, new_params):
        new_params[j] = saved_state_dict[i]

    network.load_state_dict(new_params)
    for layer in lr_layers:
        print(layer.r)

    if args.wandb == 1:
        import wandb

        watermark = "lr_{}-target_{}_surrogate_{}_data-{}".format(
            args.wandb_tag,
            args.target_network,
            args.surrogate_network,
            args.dataID,
        )

        wandb.init(
            project="lr_{}-target{}_surrogate{}_data-{}".format(
                args.wandb_tag,
                args.target_network,
                args.surrogate_network,
                args.dataID,
            ),
        )
        wandb.config.update(args)
        wandb.watch(network)

    # network = torch.nn.DataParallel(network).cuda()
    network.eval()

    OA_clean, clean_class_acc, clean_class_names = test_acc(
        network, classname, clean_loader, 1, num_classes, print_per_batches=10,is_vit=args.target_network=="vit32b"
    )
    OA_adv, class_acc, class_names = test_acc(
        network, classname, adv_loader, 1, num_classes, print_per_batches=10,is_vit=args.target_network=="vit32b"
    )
    print("Clean Test Set OA:", OA_clean * 100)
    print(args.attack_func + " Test Set OA:", OA_adv * 100)

    if args.wandb == 1:

        wandb.log(
            {
                "clean_val_acc": OA_clean,
                "adv_val_acc_" + args.attack_func: OA_adv,
            }
        )
        for name, acc in zip(class_names, class_acc):
            wandb.log({"class_" + name: acc})

        for name, acc in zip(clean_class_names, clean_class_acc):
            wandb.log({"clean_class_" + name: acc})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataID", type=int, default=1)
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./dataset/data_adversarial_rs/",
        help="dataset path.",
    )

    parser.add_argument(
        "--target_network",
        type=str,
        default="vgg16",
        help="alexnet,vgg11,vgg16,vgg19,inception,resnet18,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d,densenet121,densenet169,densenet201,regnet_x_400mf,regnet_x_8gf,regnet_x_16gf",
    )
    parser.add_argument(
        "--surrogate_network",
        type=str,
        default="vgg16",
        help="alexnet,vgg11,vgg16,vgg19,inception,resnet18,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d,densenet121,densenet169,densenet201,regnet_x_400mf,regnet_x_8gf,regnet_x_16gf",
    )
    parser.add_argument("--save_path_prefix", type=str, default="./models/")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--attack_func",
        type=str,
        default="fgsm",
        help="fgsm,ifgsm,cw,tpgd,jitter,mixup,mixcut",
    )
    # Low rank parameters
    parser.add_argument(
        "--rmax", type=float, default=200
    )  # this must match the max rank of the trained model (Needs to be made nicer later)
    parser.add_argument("--tol", type=float, default=0.075)
    parser.add_argument("--robusteness_regularization_beta", type=float, default=0.05)
    parser.add_argument("--init_r", type=float, default=50)
    parser.add_argument("--wandb", type=int, default=1)
    parser.add_argument(
        "--attack_epsilon", type=float, default=1
    )  # only for wandb registry
    parser.add_argument("--wandb_tag", type=str, default="model")

    main(parser.parse_args())
