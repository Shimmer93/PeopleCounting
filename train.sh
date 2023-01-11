#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/utils
# python dmap_gen.py --path /mnt/home/zpengac/USERDIR/Crowd_counting/datasets/vidcrowd2
# cd /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting
# python main_temporal.py fit --config configs/vidcrowd_vcformer.yml
#python main.py fit --config configs/vidcrowd_csrnet.yml
#python main.py fit --config configs/vidcrowd_sacanet.yml
# python main.py fit --config configs/vidcrowd_mcnn.yml
#python main.py fit --config configs/vidcrowd_bl.yml
# python main.py fit --config configs/vidcrowd_man.yml
# python main.py fit --config configs/vidcrowd_msfanet.yml
# python main.py fit --config configs/jhu_sasnet.yml
# python main_smap.py fit --config configs/jhu_swinsdcnet2.yml
python main_smap.py test --config configs/jhu_swinsdcnet3.yml --ckpt_path /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_swinsdcnet_base_newdmap_300epoch/checkpoints/epoch=190_mse=12793.58_mae=46.98_nae=0.18.ckpt
# python main_smap.py test --config configs/jhu_swinsdcnet.yml --ckpt_path /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_swinsdcnet_3_3_300epoch/checkpoints/epoch=225_mse=17057.05_mae=48.74_nae=0.20.ckpt
# python main_smap.py test --config configs/jhu_swinsdcnetnew.yml --ckpt_path /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_swinsdcnetnew_log100_300epoch_add/checkpoints/epoch=204_mse=16487.44_mae=51.40_nae=0.23.ckpt
# python main.py test --config configs/jhu_man.yml
# python main_smap.py test --config configs/jhu_swinsdcnet.yml --ckpt_path /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_swinsdcnet_newdmap_300epoch/checkpoints/epoch=254_mse=16087.36_mae=48.24_nae=0.19.ckpt
# python main_smap.py test --config configs/jhu_sdcnet.yml --ckpt_path /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_sdcnet_8scale_1e4_halfteacherforcing/checkpoints/epoch=094_mse=71872.04_mae=64.90.ckpt
# python main_temporal.py fit --config configs/vidcrowd_temporalcsrnet.yml