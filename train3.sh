#!/bin/bash

CUDA_VISIBLE_DEVICES=1

# python main.py fit --config configs/vidcrowd_cctrans.yml
# python main_smap.py fit --config configs/jhu_scalecount.yml
# python main_smap.py fit --config configs/jhu_sdcnet.yml
python main_smap.py test --config configs/jhu_sdcnet.yml --ckpt_path /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_sdcnet_8scale_1e4_300epoch/checkpoints/epoch=273_mse=41471.65_mae=53.31.ckpt
# python main_smap.py fit --config configs/jhu_sdcnet_bayesian.yml
# python main_temporal.py fit --config configs/vidcrowd_vcformer.yml
# python main_temporal.py fit --config configs/vidcrowd_temporalcsrnet.yml