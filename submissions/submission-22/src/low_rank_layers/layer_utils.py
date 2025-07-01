from torch import nn
from src.low_rank_layers.lr_layer_augBUG import LowRankLayerAugBUG
from src.low_rank_layers.lr_layer_augBUG_conv2d import Conv2dLowRankLayerAugBUG
from src.low_rank_layers.custom_linear import CustomLinearLayer
from src.low_rank_layers.custom_conv2d import CustomConv2d


def replace_layer_by_name(model, layer_name, max_rank, init_rank, tol):
    """
    Replaces a submodule in a PyTorch model by its hierarchical name.

    Args:
        model (nn.Module): The top-level PyTorch module.
        layer_name (str): The dot-separated hierarchical name of the layer to replace.
        custom_layer_constructor (callable): Constructor function for the new layer.
            Takes `original_layer` and `rmax` as arguments.
        max_rank (int): The max rank parameter for the custom layer.

    Returns:
        None: The function modifies the model in-place.
    """
    # Split the hierarchical name into components
    name_parts = layer_name.split(".")

    # Traverse the hierarchy up to the parent of the target submodule
    submodule = model
    for part in name_parts[:-1]:
        if not hasattr(submodule, part):
            raise ValueError(f"Submodule '{part}' not found in model.")
        submodule = getattr(submodule, part)

    # Retrieve the target submodule
    target_name = name_parts[-1]
    if not hasattr(submodule, target_name):
        raise ValueError(f"Layer '{target_name}' not found in submodule.")

    # Get the original layer
    original_layer = getattr(submodule, target_name)

    # Replace the target submodule with the custom layer
    lr_layer = LowRankLayerAugBUG(
        in_features=original_layer.in_features,
        out_features=original_layer.out_features,
        bias=True,
        rmax=max_rank,
        rmin=2,
        init_rank=init_rank,
        tol=tol,
        original_layer=original_layer,
    )

    setattr(
        submodule,
        target_name,
        lr_layer,
    )
    return lr_layer


# Function to transform the model
def transform_to_low_rank(model, max_rank=None, init_rank=None, tol=None):
    """
    Recursively transforms a model by replacing Conv2d and Linear layers with custom low-rank layers.

    Args:
        model (nn.Module): The input PyTorch model.
        max_rank (int): Maximum rank for the low-rank approximation.
        init_rank (int): Initial rank for the low-rank approximation.
        tol (float): Tolerance for the low-rank approximation.

    Returns:
        transformed_model (nn.Module): The transformed model with low-rank layers.
        lr_layers (list): List of all replaced low-rank layers.
    """
    lr_layers = []
    layers_to_replace = []

    def list_layers(module):
        for name, layer in module.named_children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layers_to_replace.append((name, layer, module))
            elif len(list(layer.children())) > 0:
                # Recursively traverse nested modules
                list_layers(layer)

    def replace_layers_in_list(
        num_classes, exclude_last_layer=True, exclude_first_layer=True
    ):
        if not exclude_first_layer:
            name, layer, module = layers_to_replace[0]
            if isinstance(layer, nn.Conv2d):
                new_layer = CustomConv2d(
                    in_planes=layer.in_channels,
                    out_planes=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    groups=layer.groups,
                    dilation=layer.dilation,
                    original_layer=layer,
                )
                setattr(module, name, new_layer)
                lr_layers.append(new_layer)
            else:
                exit("Did not expect non-conv layer in first layer")
        layers_to_replace.pop(0)  # remove last layer from "todo list"
        if not exclude_last_layer:
            name, layer, module = layers_to_replace[-1]
            if isinstance(layer, nn.Linear):
                # Replace Linear with LRLinear
                new_layer = CustomLinearLayer(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    original_layer=layer,
                )
                setattr(module, name, new_layer)
                lr_layers.append(new_layer)
            else:
                exit("Did not expect non-linear layer in last layer")
        layers_to_replace.pop(-1)  # remove last layer from "todo list"

        for name, layer, module in layers_to_replace:
            if name in ["query", "key", "value"]:
                # Skip these layers
                continue
            
            if isinstance(layer, nn.Conv2d):
                # Replace Conv2d with LRConv2d
                # pint("conv2d")
                new_layer = Conv2dLowRankLayerAugBUG(
                    in_planes=layer.in_channels,
                    out_planes=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    groups=layer.groups,
                    dilation=layer.dilation,
                    rmax=max_rank,
                    rmin=2,
                    init_rank=init_rank,
                    tol=tol,
                    original_layer=layer,
                )
                setattr(module, name, new_layer)
                lr_layers.append(new_layer)
            elif isinstance(layer, nn.Linear):
                # Replace Linear with LRLinear
                new_layer = LowRankLayerAugBUG(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    bias=True,
                    rmax=max_rank,
                    rmin=2,
                    init_rank=init_rank,
                    tol=tol,
                    original_layer=layer,
                )
                setattr(module, name, new_layer)
                lr_layers.append(new_layer)

    """
    def replace_layers(module, exclude_last_layer=True):

        for name, layer in module.named_children():
            if isinstance(layer, nn.Conv2d):
                # Replace Conv2d with LRConv2d
                # pint("conv2d")
                new_layer = Conv2dLowRankLayerAugBUG(
                    in_planes=layer.in_channels,
                    out_planes=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    groups=layer.groups,
                    dilation=layer.dilation,
                    rmax=max_rank,
                    rmin=10,
                    init_rank=init_rank,
                    tol=tol,
                    original_layer=layer,
                )
                setattr(module, name, new_layer)
                lr_layers.append(new_layer)
            elif isinstance(layer, nn.Linear):
                # Replace Linear with LRLinear
                new_layer = LowRankLayerAugBUG(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    bias=True,
                    rmax=max_rank,
                    rmin=10,
                    init_rank=init_rank,
                    tol=tol,
                    original_layer=layer,
                )
                setattr(module, name, new_layer)
                lr_layers.append(new_layer)
            elif len(list(layer.children())) > 0:
                # Recursively traverse nested modules
                replace_layers(layer)
    """

    list_layers(model)
    # Start replacing layers from the top-level module
    replace_layers_in_list(model, exclude_last_layer=False, exclude_first_layer=False)

    return model, lr_layers
