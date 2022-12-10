#!/bin/bash

CUDA_VISIBLE_DEVICES=2

# python main.py fit --config configs/vidcrowd_cctrans.yml
# python main_smap.py fit --config configs/jhu_scalecount.yml
python main_smap.py fit --config configs/jhu_sdcnet.yml
# python main_temporal.py fit --config configs/vidcrowd_vcformer.yml
# python main_temporal.py fit --config configs/vidcrowd_temporalcsrnet.yml