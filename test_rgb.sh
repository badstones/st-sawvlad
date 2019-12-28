#!/bin/bash

num_centers=64
test_segments=1
seqvlad_type=seqvlad
#seqvlad_type=bidirect
#seqvlad_type=unshare_bidirect
timesteps=6
split=01
pref=${seqvlad_type}_t${timesteps}_"split${split}"
python test_models.py ucf101 RGB /root/data/ucf101_script/mo_test${split}.txt \
    ./models/rgb/ucf_split01_rgb_actionvlad_model_best.pth.tar \
    --arch BNInception \
    --save_scores ./results/rgb/seqvlad_rgb_k${num_centers}_s${test_segments}_${pref} \
    --num_centers ${num_centers} \
    --timesteps ${timesteps} \
    --redu_dim 512 \
    --sources /root/data/UCF-101_rgb_flow/ \
    --activation softmax \
    --seqvlad_type ${seqvlad_type} \
    --test_segments ${test_segments}
