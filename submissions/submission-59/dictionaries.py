DATASET2IMAGE_COLUMN = {
    "mnist": "image",
    "fashion-mnist": "image",
    "imagenet-1k": "image",
    "cifar10": "img",
    "cifar100": "img",
    "cifar100-fine": "img",
    "cifar100-coarse": "img",
}

DATASET2LABEL_COLUMN = {
    "mnist": "label",
    "fashion-mnist": "label",
    "imagenet-1k": "label",
    "cifar10": "label",
    "cifar100": "fine_label",
    "cifar100-fine": "fine_label",
    "cifar100-coarse": "coarse_label",
}

DATASET2NUM_CLASSES = {
    "mnist": 10,
    "fashion-mnist": 10,
    "imagenet-1k": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "cifar100-fine": 100,
    "cifar100-coarse": 20,
}

MODEL_NAME2HF_NAME = {
    "vit-small-patch16-224": "WinKawaks/vit-small-patch16-224",
    "deit-small-patch16-224": "facebook/deit-small-patch16-224",
    "dinov2-small": "facebook/dinov2-small",
    "resnet-34": "microsoft/resnet-18",
    "clip-base": "openai/clip-vit-base-patch32",
}

DATASET_NAME2HF_NAME = {
    "mnist": "mnist",
    "fashion-mnist": "zalando-datasets/fashion_mnist",
    "imagenet-1k": "ILSVRC/imagenet-1k",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "cifar100-fine": "cifar100",
    "cifar100-coarse": "cifar100",
}

MODEL2NUM_LAYERS = {
    "WinKawaks/vit-small-patch16-224": 12,
    "WinKawaks/vit-tiny-patch16-224": 12,
    "facebook/deit-small-patch16-224": 12,
    "microsoft/beit-base-patch16-224": 12,
    "google/vit-base-patch16-224": 12,
    "google/vit-large-patch16-224": 24,
    "facebook/dinov2-small": 12,
    "facebook/dinov2-base": 12,
    "microsoft/resnet-34": 16,
    "openai/clip-vit-base-patch32": 12,
}

MODEL2CONFIGS = {
    "facebook/dinov2-small": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": None,
        "layers_accept_masks": False,
    },
    "facebook/dinov2-base": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": None,
        "layers_accept_masks": False,
    },
    "google/vit-base-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "google/vit-large-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "facebook/deit-small-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "WinKawaks/vit-small-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "WinKawaks/vit-tiny-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "openai/clip-vit-base-patch32": {
        "embeddings_path": "vision_model.embeddings",
        "layers_parent_path": "vision_model.encoder",
        "layers_attribute_name": "layers",  # Note 'layers'
        "pre_norm_path": "vision_model.pre_layrnorm",
        "post_norm_path": "vision_model.post_layernorm",
        "pooler_path": None,
        "layers_accept_masks": True,
    },
    "open_clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K": {
        "embeddings_path": "visual.conv1",
        "layers_parent_path": "visual.transformer",
        "layers_attribute_name": "resblocks",
        "pre_norm_path": "visual.ln_pre",
        "post_norm_path": "visual.ln_post",
        "pooler_path": None,
        "layers_accept_masks": True,
        "needs_conv1_processing": True,
        "class_embedding_path": "visual.class_embedding",
        "positional_embedding_path": "visual.positional_embedding",
        "embedding_dropout_path": "visual.patch_dropout",
    },
}
