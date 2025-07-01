import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def save_cifar10_images(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    image_dir = os.path.join(base_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    transform = transforms.Compose(
        [transforms.ToTensor()]  # transforms.Resize((256, 256)),
    )

    train_set = torchvision.datasets.CIFAR10(
        root=base_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=base_dir, train=False, download=True, transform=transform
    )

    classes = train_set.classes
    class_dirs = {}
    class_dirs_test = {}

    for class_name in classes:
        class_dir = os.path.join(image_dir, "train")
        class_dir_test = os.path.join(image_dir, "test")
        class_dir = os.path.join(class_dir, class_name)
        class_dir_test = os.path.join(class_dir_test, class_name)

        os.makedirs(class_dir, exist_ok=True)
        os.makedirs(class_dir_test, exist_ok=True)
        class_dirs[class_name] = class_dir
        class_dirs_test[class_name] = class_dir_test

    def save_images(dataset, txt_filename, train):
        txt_filepath = os.path.join(base_dir, txt_filename)

        with open(txt_filepath, "w") as f:
            class_counts = {class_name: 0 for class_name in classes}
            for img, label in tqdm(dataset, desc=f"Processing {txt_filename}"):
                class_name = classes[label]
                img_filename = f"{class_name}{class_counts[class_name]}.tif"
                if train:
                    img_path = os.path.join(class_dirs[class_name], img_filename)
                else:
                    img_path = os.path.join(class_dirs_test[class_name], img_filename)

                img_pil = transforms.ToPILImage()(img)  # Convert back to PIL for saving
                img_pil.save(img_path, format="TIFF")
                f.write(f"{img_path} {label}\n")
                class_counts[class_name] += 1

    save_images(train_set, "../../Cifar10_train.txt", train=True)
    save_images(test_set, "../../Cifar10_test.txt", train=False)


def sort_text_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    lines.sort()
    with open(filepath, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    base_directory = "dataset/data_adversarial_rs/Cifar10"  # Change this path as needed
    save_cifar10_images("dataset/data_adversarial_rs/Cifar10")
