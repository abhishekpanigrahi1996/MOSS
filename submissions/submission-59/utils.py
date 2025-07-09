from functools import partial, reduce
import torch
from torch import nn
from typing import Optional, Sequence, List, Dict
from collections import defaultdict
from pytorch_lightning import seed_everything
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(obj, path: Optional[str]):
    """Gets an attribute/module using a dot-separated path string."""
    if not path:
        return None
    try:
        return reduce(getattr, path.split("."), obj)
    except AttributeError:
        print(f"Warning: Could not resolve path '{path}' in object {obj.__class__.__name__}. Returning None.")
        return None


@torch.no_grad()
def image_encode(
    samples: Sequence[Dict],
    processor,
    image_name,
    label_name,
):

    images: List[torch.Tensor] = [sample[image_name].convert("RGB") for sample in samples]
    images: List[torch.Tensor] = [processor(images=image, return_tensors="pt")["pixel_values"] for image in images]

    images = torch.cat(images, dim=0)

    return {"images": images, "labels": torch.tensor([sample[label_name] for sample in samples])}


@torch.no_grad()
def open_clip_image_encode(batch, processor, image_name="image", label_name="label"):
    images = [item[image_name] for item in batch]

    pixel_values = torch.stack([processor(img) for img in images])
    labels = torch.tensor([item[label_name] for item in batch])

    return {"pixel_values": pixel_values, "labels": labels, "attention_mask": None}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert_parameters(num_parameters):
    if num_parameters >= 1_000_000_000:
        return f"{num_parameters / 1_000_000_000:.2f}B"  # Billions
    elif num_parameters >= 1_000_000:
        return f"{num_parameters / 1_000_000:.2f}M"  # Millions
    elif num_parameters >= 1_000:
        return f"{num_parameters / 1_000:.2f}K"  # Thousands
    else:
        return str(num_parameters)


@torch.no_grad()
def extract_representations(
    encoder: nn.Module,
    max_samples: int,
    loader: torch.utils.data.DataLoader,
    model_config: dict,
    model_is_open_clip: bool,
    seed: int = 0,
):
    seed_everything(seed)
    encoder.eval().to(device)

    stored_samples = 0
    layer_outputs_batches = defaultdict(list)
    hooks = []

    try:
        if model_is_open_clip:
            print("Using hooks to extract intermediate layers.")
            layers_parent_module = resolve_path(encoder, model_config["layers_parent_path"])
            if layers_parent_module is None:
                raise ValueError(f"Could not resolve layers_parent_path: {model_config['layers_parent_path']}")

            layers_list = getattr(layers_parent_module, model_config["layers_attribute_name"])
            if not isinstance(layers_list, nn.ModuleList) and not isinstance(layers_list, list):
                raise TypeError(
                    f"Expected ModuleList or list at {model_config['layers_parent_path']}.{model_config['layers_attribute_name']}, found {type(layers_list)}"
                )

            num_layers = len(layers_list)
            print(f"Found {num_layers} layers to hook.")

            def get_output_hook(module, input, output, layer_idx):
                hidden_state = output[0] if isinstance(output, tuple) else output
                current_batch_size = hidden_state.shape[0]
                samples_needed_from_batch = max(0, min(current_batch_size, max_samples - stored_samples))

                if samples_needed_from_batch > 0:
                    layer_outputs_batches[layer_idx].append(
                        hidden_state[:samples_needed_from_batch].cpu().detach().clone()
                    )

            for i, layer_module in enumerate(layers_list):
                hook = layer_module.register_forward_hook(partial(get_output_hook, layer_idx=i))
                hooks.append(hook)

            for batch in tqdm(loader, desc="Extracting via Hooks"):
                if stored_samples >= max_samples:
                    break

                image_input = batch.get("pixel_values", batch.get("images"))
                if image_input is None:
                    print("Warning: Batch missing 'pixel_values' or 'images'. Skipping.")
                    continue
                image_input = image_input.to(device)
                batch_size = image_input.shape[0]

                if hasattr(encoder, "encode_image"):
                    _ = encoder.encode_image(image_input)
                elif hasattr(encoder, "visual"):
                    _ = encoder.visual(image_input)
                else:
                    _ = encoder(image_input)

                stored_samples += min(batch_size, max_samples - stored_samples)

        else:
            if not hasattr(encoder, "config") or not hasattr(encoder.config, "num_hidden_layers"):
                raise ValueError("Cannot determine number of layers. Model lacks standard config.")

            num_layers = encoder.config.num_hidden_layers

            for batch in tqdm(loader, desc="Extracting via HiddenStates"):
                if stored_samples >= max_samples:
                    break

                image_input = batch.get("pixel_values", batch.get("images"))
                if image_input is None:
                    print("Warning: Batch missing 'pixel_values' or 'images'. Skipping.")
                    continue
                image_input = image_input.to(device)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                outputs = encoder(
                    pixel_values=image_input,
                    # attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                    raise ValueError("Model did not return 'hidden_states'. Ensure 'output_hidden_states=True'.")

                actual_hidden_states = outputs.hidden_states[1:]

                if len(actual_hidden_states) != num_layers:
                    print(
                        f"Warning: Expected {num_layers} hidden states after layers, but got {len(actual_hidden_states)}"
                    )

                num_to_add = min(image_input.size(0), max_samples - stored_samples)

                if num_to_add > 0:
                    for layer_idx, layer_output in enumerate(actual_hidden_states):
                        layer_outputs_batches[layer_idx].append(layer_output[:num_to_add].cpu().detach().clone())
                    stored_samples += num_to_add

    finally:
        if hooks:
            for hook in hooks:
                hook.remove()

    final_layer_embeddings = {}
    captured_layers = sorted(layer_outputs_batches.keys())
    for layer_idx in captured_layers:
        if layer_outputs_batches[layer_idx]:
            concatenated = torch.cat(layer_outputs_batches[layer_idx], dim=0)
            final_layer_embeddings[layer_idx] = concatenated[:max_samples]
            # print(f"  Layer {layer_idx}: Final shape {final_layer_embeddings[layer_idx].shape}")
        else:
            print(f"  Layer {layer_idx}: No embeddings captured.")

    if not final_layer_embeddings:
        print("Warning: No intermediate layer embeddings were captured.")
    elif len(final_layer_embeddings) == 0:
        print(f"Warning: Captured embeddings for {len(final_layer_embeddings)} layers. Check extraction logic.")

    return final_layer_embeddings


def extract_specific_layers(model, max_samples, dataloader, layers_to_skip, seed=0):
    seed_everything(seed)

    stored_samples = 0
    layer_embeddings = defaultdict(list)

    model.to(device)
    model.eval()

    layers_to_skip_indices = {layer for pair in layers_to_skip for layer in pair}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)

            outputs = model(images)
            hidden_states = outputs.hidden_states[1:]

            num_to_add = min(max_samples - stored_samples, images.size(0))

            if num_to_add > 0:
                for layer_idx, layer_output in enumerate(hidden_states):
                    if layer_idx in layers_to_skip_indices:
                        layer_embeddings[layer_idx].append(layer_output[:num_to_add].cpu())

                stored_samples += num_to_add

            if stored_samples >= max_samples:
                break

    for layer_idx in layer_embeddings:
        layer_embeddings[layer_idx] = torch.cat(layer_embeddings[layer_idx], dim=0)

    flattened_layers = torch.flatten(torch.tensor(layers_to_skip))
    unique_elements = torch.tensor(list(set(flattened_layers.tolist())))

    assert len(layer_embeddings) == len(unique_elements)

    return layer_embeddings


def extract_all_layers(model, max_samples, dataloader, only_cls):
    stored_samples = 0

    layer_embeddings = defaultdict(list)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)

            outputs = model(images)
            hidden_states = outputs.hidden_states[1:]

            num_to_add = min(max_samples - stored_samples, images.size(0))

            if num_to_add > 0:
                for layer_idx, layer_output in enumerate(hidden_states):
                    if only_cls:
                        layer_embeddings[layer_idx].append(layer_output[:num_to_add, 0, :].cpu())
                    else:
                        layer_embeddings[layer_idx].append(layer_output[:num_to_add].cpu())
                stored_samples += num_to_add

            if stored_samples >= max_samples:
                break

    for layer_idx in layer_embeddings:
        layer_embeddings[layer_idx] = torch.cat(layer_embeddings[layer_idx], dim=0)

    # assert len(layer_embeddings) == 12

    return layer_embeddings


def extract_resnet_blocks(model, dataloader, max_samples: int = 1000, extract_every_n: int = 1):

    activations = defaultdict(list)

    def hook_fn(block_idx):
        def hook(module, input, output):
            activations[block_idx].append(output.detach().cpu())

        return hook

    hooks = []
    block_idx = 0
    for stage_idx, stage in enumerate(model.encoder.stages):
        for layer_idx, layer in enumerate(stage.layers):
            if layer_idx % extract_every_n == 0:
                hooks.append(layer.register_forward_hook(hook_fn(block_idx)))
                block_idx += 1

    stored_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            num_to_add = min(max_samples - stored_samples, images.size(0))

            if num_to_add > 0:
                model(images[:num_to_add])
                stored_samples += num_to_add

            if stored_samples >= max_samples:
                break

    for hook in hooks:
        hook.remove()

    for block_idx in activations:
        activations[block_idx] = torch.cat(activations[block_idx], dim=0)

    return activations
