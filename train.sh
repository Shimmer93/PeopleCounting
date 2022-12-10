#!/bin/bash

CUDA_VISIBLE_DEVICES=1
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
python main_smap.py fit --config configs/jhu_sdcnet.yml
# python main_temporal.py fit --config configs/vidcrowd_temporalcsrnet.yml