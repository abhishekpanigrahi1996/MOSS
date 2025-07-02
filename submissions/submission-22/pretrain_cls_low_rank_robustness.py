import os
import time
import torch
import argparse
from torch import nn
from torch.utils import data
from torchvision import transforms
from src.tools.utils import *
import src.tools.model as models
from dataset.scene_dataset import *
from src.utils.io_utils import create_csv_files, append_singular_values

from src.low_rank_layers.layer_utils import transform_to_low_rank
from src.utils.data_utils import choose_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR


import tensorly as tly

tly.set_backend("pytorch")


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
        raise NotImplementedError

    print_per_batches = args.print_per_batches
    save_path_prefix = (
        args.save_path_prefix + DataName + "/Pretrain/low_rank/" + args.network + "/"
    )

    if os.path.exists(save_path_prefix) == False:
        os.makedirs(save_path_prefix)

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
        train_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_train.txt",
                transform=composed_transforms,
            ),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_loader = data.DataLoader(
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

        train_transform = T.Compose(
            [
                T.Resize(size=(args.crop_size, args.crop_size)),
                T.RandomCrop(
                    size=args.crop_size,
                    padding=4,
                    padding_mode="reflect",
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(*stats, inplace=True),
            ]
        )
        valid_transform = T.Compose(
            [
                T.Resize(size=(args.crop_size, args.crop_size)),
                T.ToTensor(),
                T.Normalize(*stats),
            ]
        )

        train_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_train.txt",
                transform=train_transform,
            ),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_loader = data.DataLoader(
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

    ###################Network Definition###################
    print("Network: %s" % args.network)
    if args.network == "alexnet":
        Model = models.alexnet(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif args.network == "vgg11":
        Model = models.vgg11(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif args.network == "vgg16":
        Model = models.vgg16(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif args.network == "vgg19":
        Model = models.vgg19(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif args.network == "inception":
        Model = models.inception_v3(pretrained=True, aux_logits=False)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "resnet18":
        Model = models.resnet18(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "resnet34":
        Model = models.resnet34(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "resnet50":
        Model = models.resnet50(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "resnet101":
        Model = models.resnet101(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "resnext50_32x4d":
        Model = models.resnext50_32x4d(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "resnext101_32x8d":
        Model = models.resnext101_32x8d(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "densenet121":
        Model = models.densenet121(pretrained=True)
        Model.classifier = nn.Linear(1024, num_classes)
    elif args.network == "densenet169":
        Model = models.densenet169(pretrained=True)
        Model.classifier = nn.Linear(1664, num_classes)
    elif args.network == "densenet201":
        Model = models.densenet201(pretrained=True)
        Model.classifier = nn.Linear(1920, num_classes)
    elif args.network == "regnet_x_400mf":
        Model = models.regnet_x_400mf(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "regnet_x_8gf":
        Model = models.regnet_x_8gf(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "regnet_x_16gf":
        Model = models.regnet_x_16gf(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network == "vit32b":
        from src.tools.model_vit import vit_b32

        Model = vit_b32(num_classes)

    Model = torch.nn.DataParallel(Model).cuda()

    Model, lr_layers = transform_to_low_rank(
        Model, max_rank=args.rmax, init_rank=args.init_r, tol=args.tol
    )
    # exit(1)
    print(Model)

    if args.load_model != 0:  # continue training a pre-trained low-rank model.
        model_name = args.load_model_name + ".pth"
        print(
            "Loading model from {}".format(os.path.join(save_path_prefix, model_name))
        )
        saved_state_dict = torch.load(os.path.join(save_path_prefix, model_name))
        new_params = Model.state_dict().copy()

        for i, j in zip(saved_state_dict, new_params):
            new_params[j] = saved_state_dict[i]

        Model.load_state_dict(new_params)
        print("--- Model structure ----")
        print(Model)
        print("--- Initial ranks ----")
        for layer in lr_layers:
            print(layer.r)
        print("--- Initial ranks ----")

        Model.load_state_dict(new_params)

    if args.wandb == 1:
        import wandb

        wandb.init(
            project="{}_model_lr-{}_data-{}".format(
                args.wandb_tag,
                args.network,
                args.dataID,
            ),
        )
        wandb.config.update(args)
        wandb.watch(Model)

    Model_optimizer = torch.optim.Adam(
        Model.parameters(), lr=args.lr
    )  # , weight_decay=args.weight_decay

    Model_optimizer_ft = torch.optim.Adam(
        Model.parameters(), lr=args.lr
    )  # , weight_decay=args.weight_decay

    # Calculate the number of batches
    num_batches = len(train_loader)
    # Initialize cosine annealing learning rate schedulers
    scheduler = CosineAnnealingLR(
        Model_optimizer,
        T_max=(args.num_epochs + args.num_epochs_low_rank_ft) * num_batches,
    )
    scheduler_ft = CosineAnnealingLR(
        Model_optimizer_ft,
        T_max=(args.num_epochs + args.num_epochs_low_rank_ft) * num_batches,
    )

    num_batches = len(train_loader)

    cls_loss = torch.nn.CrossEntropyLoss()
    num_steps = args.num_epochs * num_batches
    num_steps_ft = args.num_epochs_low_rank_ft * num_batches
    hist = np.zeros((num_steps + num_steps_ft, 3))
    index_i = -1

    augmented = False
    log_file_path = (
        args.log_path
        + "/"
        + DataName
        + "/Pretrain/low_rank/"
        + args.network
        + "/beta_"
        + str(args.robusteness_regularization_beta)
        + "/"
        + "/tol_"
        + str(args.tol)
        + "/"
    )
    create_csv_files(
        n=len(lr_layers),
        root_dir=log_file_path,
    )  # create log files
    iteration = 0

    print("Start low-rank training")

    for epoch in range(args.num_epochs):
        for batch_index, src_data in enumerate(train_loader):

            iteration += 1

            index_i += 1

            tem_time = time.time()
            Model.train()
            Model_optimizer.zero_grad()

            X_train, Y_train, _ = src_data

            X_train = X_train.cuda()
            Y_train = Y_train.cuda()

            if args.network == "vit32b":
                output = Model(X_train).logits
            else:
                _, output = Model(X_train)
            # CE Loss
            _, src_prd_label = torch.max(output, 1)
            cls_loss_value = cls_loss(output, Y_train)
            if args.robusteness_regularization_beta > 0:
                for layer in lr_layers:
                    cls_loss_value += layer.robustness_regularization(
                        beta=args.robusteness_regularization_beta
                    )
            cls_loss_value.backward()

            # ----- DLRT -----
            # 1) Augment
            if (
                batch_index % args.num_local_iter
                == 0
                # and epoch < args.num_epochs - 2  # no rank adaptation in the last epochs
            ):
                augmented = True
                # print("augment")
                for layer in lr_layers:
                    layer.augment(Model_optimizer)
            else:
                # 2) Train
                for layer in lr_layers:
                    layer.set_basis_grad_zero()
                Model_optimizer.step()  # This would be the standard training step

            # 3) Truncate
            if (
                batch_index % args.num_local_iter
                == args.num_local_iter - 1
                # and epoch < args.num_epochs - 2  # no rank adaptation in the last epochs
            ):
                augmented = False
                # print("truncate")
                for layer in lr_layers:
                    layer.truncate(Model_optimizer)
            elif (
                augmented and batch_index == len(train_loader) - 1
            ):  # make sure not end epoch on augmented network
                augmented = False
                for layer in lr_layers:
                    layer.truncate(Model_optimizer)

            # ----- DLRT -----
            # Step the scheduler
            scheduler.step()
            scheduler_ft.step()

            hist[index_i, 0] = time.time() - tem_time
            hist[index_i, 1] = cls_loss_value.item()
            hist[index_i, 2] = torch.mean((src_prd_label == Y_train).float()).item()

            tem_time = time.time()
            lr_params = 0
            full_params = 0
            for layer in lr_layers:
                lr_params += layer.compute_lr_params()
                full_params += layer.compute_dense_params()
            cr = (1 - lr_params / full_params) * 100

            if (index_i + 1) % print_per_batches == 0:
                print(
                    "Step %d; Epoch %d/%d:  %d/%d Time: %.2f cls_loss = %.3f acc = %.3f cr = %.2f\n"
                    % (
                        index_i,
                        epoch + 1,
                        args.num_epochs,
                        batch_index + 1,
                        num_batches,
                        np.mean(hist[index_i - print_per_batches + 1 : index_i + 1, 0]),
                        np.mean(hist[index_i - print_per_batches + 1 : index_i + 1, 1]),
                        np.mean(hist[index_i - print_per_batches + 1 : index_i + 1, 2]),
                        cr,
                    )
                )

            if args.wandb == 1:
                current_lr = Model_optimizer.param_groups[0]["lr"]

                wandb.log(
                    {
                        "Time": np.mean(
                            hist[index_i - print_per_batches + 1 : index_i + 1, 0]
                        ),
                        "cls_loss": np.mean(
                            hist[index_i - print_per_batches + 1 : index_i + 1, 1]
                        ),
                        "acc": np.mean(
                            hist[index_i - print_per_batches + 1 : index_i + 1, 2]
                        ),
                        "compression": cr,
                        "rank ": [lr_layer.r for lr_layer in lr_layers],
                        "learning_rate": current_lr,  # Log the learning rate
                    }
                )

    print("Finished rank adaptive training, start finetuning")

    for epoch in range(args.num_epochs_low_rank_ft):
        for batch_index, src_data in enumerate(train_loader):
            index_i += 1

            tem_time = time.time()
            Model.train()
            Model_optimizer_ft.zero_grad()

            X_train, Y_train, _ = src_data

            X_train = X_train.cuda()
            Y_train = Y_train.cuda()

            if args.network == "vit32b":
                output = Model(X_train).logits
            else:
                _, output = Model(X_train)

            # CE Loss
            _, src_prd_label = torch.max(output, 1)
            cls_loss_value = cls_loss(output, Y_train)

            if args.robusteness_regularization_beta > 0:
                for layer in lr_layers:
                    cls_loss_value += layer.robustness_regularization(
                        beta=args.robusteness_regularization_beta
                    )
            cls_loss_value.backward()

            # ----- DLRT Finetuning -----
            for layer in lr_layers:
                layer.set_basis_grad_zero()

            Model_optimizer_ft.step()  # This would be the standard training step
            scheduler_ft.step()

            # ----- DLRT -----
            hist[index_i, 0] = time.time() - tem_time
            hist[index_i, 1] = cls_loss_value.item()
            hist[index_i, 2] = torch.mean((src_prd_label == Y_train).float()).item()

            tem_time = time.time()
            if (index_i + 1) % print_per_batches == 0:
                lr_params = 0
                full_params = 0
                for layer in lr_layers:
                    lr_params += layer.compute_lr_params()
                    full_params += layer.compute_dense_params()
                cr = (1 - lr_params / full_params) * 100

                print(
                    "Step %d; Epoch %d/%d:  %d/%d Time: %.2f cls_loss = %.3f acc = %.3f cr = %.2f\n"
                    % (
                        index_i,
                        epoch + 1,
                        args.num_epochs_low_rank_ft,
                        batch_index + 1,
                        num_batches,
                        np.mean(hist[index_i - print_per_batches + 1 : index_i + 1, 0]),
                        np.mean(hist[index_i - print_per_batches + 1 : index_i + 1, 1]),
                        np.mean(hist[index_i - print_per_batches + 1 : index_i + 1, 2]),
                        cr,
                    )
                )
                iteration += 1

            if args.wandb == 1:
                current_lr = Model_optimizer_ft.param_groups[0]["lr"]

                wandb.log(
                    {
                        "Time": np.mean(
                            hist[index_i - print_per_batches + 1 : index_i + 1, 0]
                        ),
                        "cls_loss": np.mean(
                            hist[index_i - print_per_batches + 1 : index_i + 1, 1]
                        ),
                        "acc": np.mean(
                            hist[index_i - print_per_batches + 1 : index_i + 1, 2]
                        ),
                        "compression": cr,
                        "rank ": [lr_layer.r for lr_layer in lr_layers],
                        "learning_rate": current_lr,  # Log the learning rate
                    }
                )
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

    print("Save Model at " + os.path.join(save_path_prefix, model_name))
    torch.save(Model.state_dict(), os.path.join(save_path_prefix, model_name))

    OA_new, _, _ = test_acc(
        Model,
        classname,
        val_loader,
        epoch + 1,
        num_classes,
        print_per_batches=10,
        is_vit=args.network == "vit32b",
    )
    if args.wandb == 1:
        wandb.log(
            {
                "val_acc_clean": OA_new,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataID", type=int, default=1)
    parser.add_argument(
        "--network",
        type=str,
        default="vgg16",
        help="alexnet,vgg11,vgg16,vgg19,inception,resnet18,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d,densenet121,densenet169,densenet201,regnet_x_400mf,regnet_x_8gf,regnet_x_16gf",
    )
    parser.add_argument("--save_path_prefix", type=str, default="./models/")
    parser.add_argument("--log_path", type=str, default="./results/")
    parser.add_argument("--load_model", type=int, default=0)
    parser.add_argument("--load_model_name", type=str, default="default_model")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./dataset/data_adversarial_rs/",
        help="dataset path.",
    )
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=0)
    parser.add_argument("--print_per_batches", type=int, default=5)
    parser.add_argument("--save_name", type=str, default="default_model")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # --- low-rank parameters ----
    parser.add_argument("--num_epochs_low_rank_ft", type=int, default=10)

    parser.add_argument("--tol", type=float, default=0.075)
    parser.add_argument("--rmax", type=float, default=200)
    parser.add_argument("--init_r", type=float, default=50)
    parser.add_argument("--num_local_iter", type=int, default=10)

    # ---- robustness regularization parameters ----
    parser.add_argument("--robusteness_regularization_beta", type=float, default=0.05)

    parser.add_argument("--wandb", type=int, default=1)
    parser.add_argument("--git", type=float, default=0.05)
    parser.add_argument("--wandb_tag", type=str, default="model")

    main(parser.parse_args())
