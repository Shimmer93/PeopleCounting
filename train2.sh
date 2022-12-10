#!/bin/bash

CUDA_VISIBLE_DEVICES=0
#python main_temporal.py fit --config configs/vidcrowd_vcformer.yml
#python main.py fit --config configs/vidcrowd_cctrans.yml
#python main.py fit --config configs/vidcrowd_mcnn.yml
#python main.py fit --config configs/vidcrowd_csrnet.yml
# python main.py fit --config configs/vidcrowd_sacanet.yml
# python main_smap.py fit --config configs/jhu_sasnet2.yml
python main_smap.py fit --config configs/jhu_sdcnet.yml
# python main_temporal.py fit --config configs/vidcrowd_lstn.yml
# python main_temporal.py fit --config configs/vidcrowd_temporalcsrnet.yml