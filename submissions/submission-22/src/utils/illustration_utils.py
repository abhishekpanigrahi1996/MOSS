from torchvision.models import resnet50, ResNet50_Weights

import torch
import matplotlib.pyplot as plt
import os
from itertools import permutations


def plot_singular_spectrum(model, output_dir="singular_spectrum"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Read all weight matrices (or tensors) from the model
    weight_matrices = []
    for name, param in model.named_parameters():
        if (
            "weight" in name and param.dim() >= 2
        ):  # Only consider weight matrices (not biases)
            weight_matrices.append((name, param.detach().cpu().numpy()))

    # Step 2 & 3: Compute SVD and plot singular values
    for param_name, weight_matrix in weight_matrices:
        # Compute the SVD
        u, s, vh = torch.linalg.svd(torch.tensor(weight_matrix), full_matrices=False)

        # Plot the singular values
        plt.figure()
        plt.plot(range(1, len(s) + 1), s, marker="o", linestyle="-")
        plt.title(f"Singular Spectrum of {param_name}")
        plt.xlabel("Index of Singular Value")
        plt.ylabel("Singular Value")
        plt.yscale("log")  # Log scale to capture wide ranges of values
        plt.grid(True)

        # Save the plot
        output_path = os.path.join(output_dir, f"{param_name}.png")
        plt.savefig(output_path)
        plt.close()

    # print(f"Singular spectrum plots saved in: {output_dir}")


def plot_singular_spectrum_and_condition_numbers(
    weight_matrices, output_dir="singular_spectrum"
):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    total_condition_product = 1  # Initialize product of condition numbers
    condition_numbers = []  # List to store condition numbers for each layer

    # Step 1: Read all weight matrices (or tensors) from the model

    # Step 2 & 3: Compute SVD, plot singular values, and compute condition numbers
    for idx, (weight_matrix, param_name) in enumerate(weight_matrices):
        # Compute the SVD
        # print("Computing SVD of layer", idx)
        # print("Shape of weight matrix:", weight_matrix.shape)
        u, s, vh = torch.linalg.svd(torch.tensor(weight_matrix), full_matrices=False)

        # Compute condition number as ratio of largest to smallest singular value
        condition_number = s.max() / s.min()
        condition_numbers.append(condition_number)  # Store the condition number
        # print(f"Condition number of {param_name} (Layer {idx}): {condition_number}")

        # Plot the singular values
        plt.figure()
        plt.plot(range(1, len(s) + 1), s, marker="o", linestyle="-")
        plt.title(f"Singular Spectrum of {param_name}")
        plt.xlabel("Index of Singular Value")
        plt.ylabel("Singular Value")
        plt.yscale("log")  # Log scale to capture wide ranges of values
        plt.grid(True)

        # Save the plot
        output_path = os.path.join(output_dir, f"{param_name}.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        # Multiply to the total condition product
        total_condition_product *= condition_number

    # Step 4: Plot condition numbers over layer index
    plt.figure()
    plt.plot(
        range(len(condition_numbers)), condition_numbers, marker="o", linestyle="-"
    )
    plt.title("Condition Numbers Over Layer Index")
    plt.xlabel("Layer Index")
    plt.ylabel("Condition Number")
    plt.yscale("log")  # Log scale for better visualization of large condition numbers
    plt.grid(True)
    plt.tight_layout()
    # Save the condition number plot
    plt.savefig(os.path.join(output_dir, "condition_numbers_over_layers.png"))
    plt.close()

    # Print the product of all condition numbers
    print(f"Product of all condition numbers: {total_condition_product}")


def get_matrices_recursive(module, flatten_dim=0, prefix=""):
    matrices = []
    # print("flatten_dim", flatten_dim)
    # Recursively traverse all submodules
    for name, layer in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name  # Track the full path name

        # If the layer itself has submodules, recurse into it
        if len(list(layer.children())) > 0 and not hasattr(layer, "S"):
            matrices.extend(
                get_matrices_recursive(layer, flatten_dim, prefix=full_name)
            )
        else:
            # Check if the layer has a weight parameter
            if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
                weight_tensor = layer.weight
                # print("======")
                # print(weight_tensor.dim())
                # print(weight_tensor.shape)
                # If tensor is a matrix (2D)
                if weight_tensor.dim() == 2:
                    matrices.append((weight_tensor, full_name + ".weight"))

                # If tensor is higher-order
                elif weight_tensor.dim() > 2:
                    # Flatten the tensor along the specified dimension
                    t = unfold_tensor(weight_tensor, mode=flatten_dim)
                    matrices.append((t, full_name + ".weight"))
                # print(matrices[-1][0].shape)
                # print("---")

            elif hasattr(layer, "S") and isinstance(layer.S, torch.Tensor):
                weight_tensor = layer.S
                rank = layer.r
                # print("======")
                # print(weight_tensor.dim())
                # print(weight_tensor.shape)
                # print("rank", rank)
                # If tensor is a matrix (2D)
                if weight_tensor.dim() == 2:
                    matrices.append((weight_tensor[:rank, :rank], full_name + ".S"))

                # If tensor is higher-order
                elif weight_tensor.dim() > 2:
                    # Flatten the tensor along the specified dimension
                    t = unfold_tensor(
                        weight_tensor[: rank[0], : rank[1], : rank[2], : rank[3]],
                        mode=flatten_dim,
                    )
                    matrices.append((t, full_name + ".S"))
                # print(matrices[-1][0].shape)
                # print("---")
    return matrices


def unfold_tensor(tensor, mode):
    # Rearrange dimensions to bring the unfolding mode to the front
    permute_order = [mode] + [i for i in range(tensor.ndim) if i != mode]
    permuted_tensor = tensor.permute(permute_order)

    # Reshape to unfold along the mode
    unfolded_tensor = permuted_tensor.reshape(tensor.shape[mode], -1)
    return unfolded_tensor


def get_condition_number(weight_matrices):
    # Ensure output directory exists

    total_condition_product = 1  # Initialize product of condition numbers
    condition_numbers = []  # List to store condition numbers for each layer

    # Step 1: Read all weight matrices (or tensors) from the model

    # Step 2 & 3: Compute SVD, plot singular values, and compute condition numbers
    for idx, (weight_matrix, param_name) in enumerate(weight_matrices):
        # Compute the SVD
        # print("Computing SVD of layer", idx)
        # print("Shape of weight matrix:", weight_matrix.shape)
        u, s, vh = torch.linalg.svd(torch.tensor(weight_matrix), full_matrices=False)

        # Compute condition number as ratio of largest to smallest singular value
        condition_number = s.max() / s.min()
        condition_numbers.append(condition_number)  # Store the condition number
        # print(f"Condition number of {param_name} (Layer {idx}): {condition_number}")

        # Multiply to the total condition product
        total_condition_product *= condition_number

    return total_condition_product, condition_numbers
