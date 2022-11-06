#!/bin/bash
nohup python train_mixmatch.py --early_stopping 10 --device 0 --num_label_raven 1000 &
nohup python train_mixmatch.py --early_stopping 10 --device 1 --num_label_raven 5000 &
nohup python train_mixmatch.py --early_stopping 10 --device 2 --num_label_raven 10000 &
nohup python train_mixmatch.py --early_stopping 10 --device 3 --num_label_raven 20000 &
wait
nohup python train_simple.py --early_stopping 10 --device 0 --num_label_raven 1000 &
nohup python train_simple.py --early_stopping 10 --device 1 --num_label_raven 5000 &
nohup python train_simple.py --early_stopping 10 --device 2 --num_label_raven 10000 &
nohup python train_simple.py --early_stopping 10 --device 3 --num_label_raven 20000 &

