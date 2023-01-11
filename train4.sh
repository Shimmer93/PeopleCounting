#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

# python main_smap.py fit --config configs/jhu_scalecount.yml
# python main_smap.py fit --config configs/jhu_scaleselectnet.yml
# python main_smap.py fit --config configs/jhu_swinsdcnet4.yml
# python main_smap.py fit --config configs/jhu_swinsdcnet3.yml
python main_smap.py fit --config configs/jhu_swinsdcnetnew.yml
# python main_smap.py fit --config configs/jhu_swinsdcnet2.yml
# python main_smap.py fit --config configs/jhu_sdcnet.yml
# python main_smap.py test --config configs/jhu_swinsdcnet3.yml --ckpt_path /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_swinsdcnet3_300epoch/checkpoints/epoch=248_mse=7915.02_mae=42.17_nae=0.28.ckpt
# python main_smap.py test --config configs/jhu_swinsdcnet.yml --ckpt_path /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_swinsdcnet_small_300epoch/checkpoints/epoch=148_mse=9562.95_mae=42.05_nae=0.22.ckpt
# python main.py fit --config configs/jhu_crossformer.yml
# python main.py fit --config configs/jhu_sacanet.yml
# python main.py fit --config configs/jhu_sasnet.yml
# python main.py test --config configs/jhu_man.yml