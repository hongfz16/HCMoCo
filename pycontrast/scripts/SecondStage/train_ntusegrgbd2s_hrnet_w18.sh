#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

set -x

num_samples=400
tag=ntusegrgbd2s_hrnet_w18_second_stage
ngpu=4
pretrain=./output/cmc_ntusegrgbd2s_model/CMCRGBD2S_HRNet_RGBD2S_Jig_False_bank_aug_C_linear_0.07_ntusegrgbd2s_official_hrnet_w18_norm_depth_v2_cosine/ckpt_epoch_100.pth
srun -p dsta \
    --job-name=ntusegrgbd2s \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --cpus-per-task=6 \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
    python -u main_contrast.py \
        --dataset NTUSeg \
        --data_folder ./data/NTURGBD \
        --train_file_list ./data/NTURGBD/NTURGBD/nturgbd_flist_clear.txt \
        --model_path ./output/cmc_ntusegrgbd2s_model \
        --tb_path ./output/cmc_ntusegrgbd2s_tb \
        --num_workers 40 \
        --epochs 100 \
        --learning_rate 0.03 \
        --lr_decay_epochs 40,50,60 \
        --batch_size 224 \
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
        --width 18 \
        --modality_missing 1 \
        --seg_root ../HRNet-Semantic-Segmentation/data/nturgbd \
        --seg_file_list ../HRNet-Semantic-Segmentation/data/nturgbd/train_list_v2.txt \
        --not_use_weighted_sampler \
        --pretrain ${pretrain} \
        --pri3d_num_samples_per_image ${num_samples} \
        --pool_method mean \
        --linear_feat_map 1 \
        --temperature 0.07 \
