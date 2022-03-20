#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

set -x

num_samples=400
tag=ntumpiirgbd2s_hrnet_w32_second_stage
ngpu=4
pretrain=./output/cmc_ntumpiirgbd2s_model/CMCRGBD2S_HRNet_RGBD2S_Jig_False_bank_aug_C_linear_0.07_ntumpiirgbd2s_official_hrnet_w32_norm_depth_cosine/ckpt_epoch_100.pth
srun -p innova \
    --job-name=ntumpiirgbd2s \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    python -u main_contrast.py \
        --dataset NTUMPII \
        --data_folder ./data/NTURGBD \
        --train_file_list ./data/NTURGBD/NTURGBD/nturgbd_flist_clear.txt \
        --model_path ./output/cmc_ntumpiirgbd2s_model \
        --tb_path ./output/cmc_ntumpiirgbd2s_tb \
        --num_workers 40 \
        --epochs 100 \
        --learning_rate 0.03 \
        --lr_decay_epochs 40,50,60 \
        --batch_size 144 \
        --method CMCJointsPri3DRGBD2S \
        --modal RGBD2S \
        --in_channel_list 3,3 \
        --nce_k 16384 \
        --nce_m 0.5 \
        --world-size 1 \
        --rank 0 \
        --multiprocessing-distributed \
        --cosine \
        --tag ${tag} \
        --arch HRNet \
        --width 32 \
        --modality_missing 1 \
        --mpii_root data/mpii/ \
        --pretrain ${pretrain} \
        --pri3d_num_samples_per_image ${num_samples} \
        --pool_method mean \
        --linear_feat_map 1 \
        --temperature 0.07 \
