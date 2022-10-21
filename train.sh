#!/bin/bash

CUDA_VISIBLE_DEVICES=1
#python main.py fit --config configs/vidcrowd_sacanet.yml
python main.py fit --config configs/vidcrowd_dssinet.yml
#python main.py fit --config configs/vidcrowd_mcnn.yml