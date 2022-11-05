#!/bin/bash
nohup python train.py --gpu 0 --num_label_raven 1000 &
nohup python train.py --gpu 1 --num_label_raven 5000 &
nohup python train.py --gpu 2 --num_label_raven 10000 &
nohup python train.py --gpu 3 --num_label_raven 20000 &
