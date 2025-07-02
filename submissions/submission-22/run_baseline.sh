run_train=true
run_attack=true

iterations="1"

regularizer_betas="0.0"

attack_f_1="condlr_fgsm"
epsilons_1="0.002 0.004 0.006"



models="vgg16" #"alexnet vgg11 vgg16 vgg19 inception resnet18 resnet50 resnet101 resnext50_32x4d resnext101_32x8d densenet121 densenet169 densenet201 regnet_x_400mf regnet_x_8gf regnet_x_16gf"
target_model="vgg16"

dataset=3
crop_size=32

#training parameters
train_batch_size=128
val_batch_size=128
num_epochs=30
weight_decay=0.0

# Enable or disable WandB logging
wandb=1 # 1 for enabled, 0 for disabled
wandb_tag="google_colab_example"

# Loop over all combinations of models and pretrained weights
for i in $(seq 1 $iterations); do
echo "iteration $i"
for model in $models; do
for beta in $regularizer_betas; do
    if [ "$run_train" = true ]; then
        python pretrain_cls_robustness.py \
            --dataID $dataset \
            --lr 1e-4 \
            --num_epochs $num_epochs \
            --network $model \
            --save_name "$beta" \
            --wandb $wandb \
            --wandb_tag "$wandb_tag" \
            --robusteness_regularization_beta "$beta" \
            --train_batch_size $train_batch_size \
            --val_batch_size $val_batch_size \
            --crop_size $crop_size \
            --weight_decay $weight_decay \
            --print_per_batches 1 \
            --root_dir ./
    fi
    if [ "$run_attack" = true ]; then
        # first attack
        for eps_attack in $epsilons_1; do
            echo "Delete all attacked images for a fresh start"
            if [ "$dataset" = "3" ]; then
                echo "Deleting ./dataset/data_adversarial_rs/Cifar10_adv/$attack_f_1/baseline/$model/*.png"
                rm ./dataset/data_adversarial_rs/Cifar10_adv/$attack_f_1/baseline/$model/*.png
            fi
            if [ "$dataset" = "2" ]; then
                echo "Deleting ./dataset/data_adversarial_rs/AID_adv/$attack_f_1/baseline/$model/*.png"
                rm ./dataset/data_adversarial_rs/AID_adv/$attack_f_1/baseline/$model/*.png
            fi
            if [ "$dataset" = "1" ]; then
                echo "Deleting ./dataset/data_adversarial_rs/UCM_adv/$attack_f_1/baseline/$model/*.png"
                rm ./dataset/data_adversarial_rs/UCM_adv/$attack_f_1/baseline/$model/*.png
            fi
          
            python attack_cls.py \
                --robusteness_regularization_beta "$beta" \
                --dataID $dataset \
                --attack_func $attack_f_1 \
                --network $model \
                --epsilon $eps_attack \
                --crop_size $crop_size \
                --save_path_prefix ./dataset/data_adversarial_rs/
            
            python test_cls.py \
                --robusteness_regularization_beta "$beta" \
                --dataID $dataset \
                --target_network $target_model \
                --surrogate_network $model \
                --attack_func $attack_f_1 \
                --wandb 1 \
                --wandb_tag "$wandb_tag" \
                --attack_epsilon $eps_attack \
                --crop_size $crop_size \
                --val_batch_size $val_batch_size \
                --root_dir ./
        done
    fi
done
done
done 