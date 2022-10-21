#!/bin/bash

CUDA_VISIBLE_DEVICES=0
python main.py fit --config configs/vidcrowd_mcnn.yml
#python main.py fit --config configs/vidcrowd_csrnet.yml
#python main.py fit --config configs/vidcrowd_sacanet.yml