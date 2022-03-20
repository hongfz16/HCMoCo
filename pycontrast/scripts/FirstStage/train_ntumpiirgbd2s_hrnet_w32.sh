#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

set -x

tag=ntumpiirgbd2s_hrnet_w32
ngpu=4
srun -p dsta \
    --job-name=ntumpiirgbdhm \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --cpus-per-task=6 \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
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
        --batch_size 160 \
        --method CMCRGBD2S \
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
        --pool_method mean \
        --IN_Pretrain pretrained_models/hrnetv2_w32_imagenet_pretrained.pth \
        --depth_Pretrain pretrained_models/hrnetv2_w32_imagenet_pretrained.pth \
