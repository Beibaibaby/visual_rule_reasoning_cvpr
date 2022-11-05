#!/bin/bash
nohup python train.py --device 0 --num_label_raven 1000 &
nohup python train.py --device 1 --num_label_raven 5000 &
nohup python train.py --device 2 --num_label_raven 10000 &
nohup python train.py --device 3 --num_label_raven 20000 &
