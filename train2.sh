#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1
#python main_temporal.py fit --config configs/vidcrowd_vcformer.yml
#python main.py fit --config configs/vidcrowd_cctrans.yml
#python main.py fit --config configs/vidcrowd_mcnn.yml
#python main.py fit --config configs/vidcrowd_csrnet.yml
python main.py fit --config configs/vidcrowd_sacanet.yml
#python main_temporal.py fit --config configs/vidcrowd_ltcrowd.yml