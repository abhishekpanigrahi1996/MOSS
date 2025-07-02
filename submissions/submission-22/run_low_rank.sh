run_train=true
run_attack=true

iterations="1" # how often you want to repeat the tests

#regularization
regularizer_betas="0.0 0.15"

attack_f_1="condlr_fgsm"
epsilons_1="0.002 0.004 0.006"


#Examples: "alexnet vgg11 vgg16 vgg19 inception resnet18 resnet50 resnet101 resnext50_32x4d resnext101_32x8d densenet121 densenet169 densenet201 regnet_x_400mf regnet_x_8gf regnet_x_16gf"
models="vgg16" 
target_model="vgg16"
dataset=3
crop_size=32


#DLRT parameters
tolerances="0.05" # Good default value
rmax="150" # Good default value
num_local_iter=10
r_init=50 # Good default value
#training parameters
train_batch_size=128
val_batch_size=128
weight_decay=0.0 #01
num_epochs=30
num_epochs_ft=1
#logging
wandb=1  # 1 for enabled, 0 for disabled
wandb_tag="google_colab_example"

# Loop over all combinations of models and pretrained weights
for i in $(seq 1 $iterations); do
echo "iteration $i"
for model in $models; do
for tol in $tolerances; do
for beta in $regularizer_betas; do
    if [ "$run_train" = true ]; then
        python pretrain_cls_low_rank_robustness.py \
            --dataID $dataset \
            --num_local_iter $num_local_iter \
            --rmax $rmax \
            --init_r $r_init \
            --lr 1e-4 \
            --num_epochs $num_epochs \
            --num_epochs_low_rank_ft $num_epochs_ft \
            --num_local_iter $num_local_iter \
            --network $model \
            --save_name "$beta" \
            --tol "$tol" \
            --wandb 1 \
            --wandb_tag "$wandb_tag" \
            --robusteness_regularization_beta "$beta" \
            --val_batch_size $val_batch_size  \
            --train_batch_size $train_batch_size \
            --crop_size $crop_size \
            --weight_decay $weight_decay \
            --print_per_batches 1 \
            --load_model 0 \
            --root_dir ./ \
            --save_path_prefix ./model_lr/
    fi
    if [ "$run_attack" = true ]; then
        for eps_attack in $epsilons_1; do
            echo "Delete all attacked images for a fresh start"

            echo "Deleting ./dataset/data_adversarial_rs/Cifar10_adv/$attack_f_1/low_rank/$model/*.png"
            rm ./dataset/data_adversarial_rs/Cifar10_adv/$attack_f_1/low_rank/$model/*.png
        
           
            python attack_cls_low_rank.py \
                --dataID $dataset \
                --rmax $rmax \
                --init_r $r_init \
                --tol "$tol" \
                --robusteness_regularization_beta "$beta" \
                --attack_func $attack_f_1 \
                --network $model \
                --crop_size $crop_size \
                --epsilon $eps_attack \
                --save_path_prefix ./ \
                --model_root_dir ./model_lr/
                         
            python test_cls_low_rank.py \
                --dataID $dataset \
                --rmax $rmax \
                --init_r $r_init \
                --tol "$tol" \
                --robusteness_regularization_beta "$beta" \
                --target_network $target_model \
                --surrogate_network $model \
                --attack_func $attack_f_1 \
                --wandb 1 \
                --wandb_tag "$wandb_tag" \
                --attack_epsilon $eps_attack \
                --crop_size $crop_size \
                --val_batch_size $val_batch_size \
                --root_dir ./ \
                --save_path_prefix ./model_lr/
        done
    fi
done
done
done
done
