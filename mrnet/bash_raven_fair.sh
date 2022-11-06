#!/bin/bash
nohup python train_supervised.py --device 1 --num_label_raven 5000 &
nohup python train_supervised.py --device 2 --num_label_raven 10000 &
nohup python train_supervised.py --device 3 --num_label_raven 20000 &
