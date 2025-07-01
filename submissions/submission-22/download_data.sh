#!/bin/bash

# Activate environment if needed (optional)
# source activate myenv

# Run the Python code
python3 - <<EOF
from dataset.data_adversarial_rs.create_cifar10 import save_cifar10_images

save_cifar10_images("dataset/data_adversarial_rs/Cifar10")
EOF
