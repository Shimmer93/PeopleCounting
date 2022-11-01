#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3
# cd /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/utils
# python dmap_gen.py --path /mnt/home/zpengac/USERDIR/Crowd_counting/datasets/vidcrowd2
# cd /mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting
#python main_temporal.py fit --config configs/vidcrowd_vcformer.yml
#python main.py fit --config configs/vidcrowd_csrnet.yml
#python main.py fit --config configs/vidcrowd_sacanet.yml
#python main.py fit --config configs/vidcrowd_mcnn.yml
#python main.py fit --config configs/vidcrowd_bl.yml
python main.py fit --config configs/vidcrowd_man.yml