#!/bin/bash

CUDA_VISIBLE_DEVICES=0
#python main_temporal.py fit --config configs/vidcrowd_vcformer.yml
#python main.py fit --config configs/vidcrowd_cctrans.yml
#python main.py fit --config configs/vidcrowd_mcnn.yml
#python main.py fit --config configs/vidcrowd_csrnet.yml
# python main.py fit --config configs/vidcrowd_sacanet.yml
# python main_smap.py fit --config configs/jhu_sasnet2.yml
python main_smap.py fit --config configs/jhu_sdcnetnew2.yml
# python main.py fit --config configs/jhu_csrnet.yml
# python main_smap.py test --config configs/jhu_sdcnet.yml --ckpt_path /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_sdcnet_6scale_1e4_halfteacherforcing/checkpoints/epoch=104_mse=53371.60_mae=63.93.ckpt
# python main_temporal.py fit --config configs/vidcrowd_lstn.yml
# python main_temporal.py fit --config configs/vidcrowd_temporalcsrnet.yml