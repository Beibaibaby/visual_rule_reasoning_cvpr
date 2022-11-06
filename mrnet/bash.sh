#!/bin/bash
nohup python train.py --device 0 --num_label_raven 1000 --save_log_path /root/autodl-tmp --dataset i-raven --path /root/autodl-tmp/RAVEN-dataset &
nohup python train.py --device 1 --num_label_raven 5000 --save_log_path /root/autodl-tmp --dataset i-raven --path /root/autodl-tmp/RAVEN-dataset &
nohup python train.py --device 2 --num_label_raven 10000 --save_log_path /root/autodl-tmp --dataset i-raven --path /root/autodl-tmp/RAVEN-dataset &
nohup python train.py --device 3 --num_label_raven 20000 --save_log_path /root/autodl-tmp --dataset i-raven --path /root/autodl-tmp/RAVEN-dataset &
