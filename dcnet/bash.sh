#!/bin/bash
nohup python train_super.py --gpu 0 --num_label_raven 3000 &
nohup python train_super.py --gpu 1 --num_label_raven 3000 &
nohup python train_super.py --gpu 2 --num_label_raven 3000 &
nohup python train_super.py --gpu 3 --num_label_raven 3000 &
wait
nohup python train_super.py --gpu 0 --num_label_raven 1000 &
nohup python train_super.py --gpu 1 --num_label_raven 1000 &
nohup python train_super.py --gpu 2 --num_label_raven 1000 &
nohup python train_super.py --gpu 3 --num_label_raven 1000 &
wait
nohup python train_mixmatch.py --gpu 0 --num_label_raven 10000 &
nohup python train_mixmatch.py --gpu 1 --num_label_raven 10000 &
nohup python train_mixmatch.py --gpu 2 --num_label_raven 10000 &
nohup python train_mixmatch.py --gpu 3 --num_label_raven 10000 &
wait
nohup python train_mixmatch.py --gpu 0 --num_label_raven 5000 &
nohup python train_mixmatch.py --gpu 1 --num_label_raven 5000 &
nohup python train_mixmatch.py --gpu 2 --num_label_raven 5000 &
nohup python train_mixmatch.py --gpu 3 --num_label_raven 5000 &
wait
nohup python train_mixmatch.py --gpu 0 --num_label_raven 3000 &
nohup python train_mixmatch.py --gpu 1 --num_label_raven 3000 &
nohup python train_mixmatch.py --gpu 2 --num_label_raven 3000 &
nohup python train_mixmatch.py --gpu 3 --num_label_raven 3000 &
wait
nohup python train_mixmatch.py --gpu 0 --num_label_raven 1000 &
nohup python train_mixmatch.py --gpu 1 --num_label_raven 1000 &
nohup python train_mixmatch.py --gpu 2 --num_label_raven 1000 &
nohup python train_mixmatch.py --gpu 3 --num_label_raven 1000 &
wait
nohup python train_simple.py --gpu 0 --num_label_raven 10000 &
nohup python train_simple.py --gpu 1 --num_label_raven 10000 &
nohup python train_simple.py --gpu 2 --num_label_raven 10000 &
nohup python train_simple.py --gpu 3 --num_label_raven 10000 &
wait
nohup python train_simple.py --gpu 0 --num_label_raven 5000 &
nohup python train_simple.py --gpu 1 --num_label_raven 5000 &
nohup python train_simple.py --gpu 2 --num_label_raven 5000 &
nohup python train_simple.py --gpu 3 --num_label_raven 5000 &
wait
nohup python train_simple.py --gpu 0 --num_label_raven 3000 &
nohup python train_simple.py --gpu 1 --num_label_raven 3000 &
nohup python train_simple.py --gpu 2 --num_label_raven 3000 &
nohup python train_simple.py --gpu 3 --num_label_raven 3000 &
wait
nohup python train_simple.py --gpu 0 --num_label_raven 1000 &
nohup python train_simple.py --gpu 1 --num_label_raven 1000 &
nohup python train_simple.py --gpu 2 --num_label_raven 1000 &
nohup python train_simple.py --gpu 3 --num_label_raven 1000 &
wait
nohup python train_drule.py --gpu 0 --num_label_raven 10000 &
nohup python train_drule.py --gpu 1 --num_label_raven 10000 &
nohup python train_drule.py --gpu 2 --num_label_raven 10000 &
nohup python train_drule.py --gpu 3 --num_label_raven 10000 &
wait
nohup python train_drule.py --gpu 0 --num_label_raven 5000 &
nohup python train_drule.py --gpu 1 --num_label_raven 5000 &
nohup python train_drule.py --gpu 2 --num_label_raven 5000 &
nohup python train_drule.py --gpu 3 --num_label_raven 5000 &
wait
nohup python train_drule.py --gpu 0 --num_label_raven 3000 &
nohup python train_drule.py --gpu 1 --num_label_raven 3000 &
nohup python train_drule.py --gpu 2 --num_label_raven 3000 &
nohup python train_drule.py --gpu 3 --num_label_raven 3000 &
wait
nohup python train_drule.py --gpu 0 --num_label_raven 1000 &
nohup python train_drule.py --gpu 1 --num_label_raven 1000 &
nohup python train_drule.py --gpu 2 --num_label_raven 1000 &
nohup python train_drule.py --gpu 3 --num_label_raven 1000 &
wait
nohup python train_drule_ws.py --gpu 0 --num_label_raven 10000 &
nohup python train_drule_ws.py --gpu 1 --num_label_raven 10000 &
nohup python train_drule_ws.py --gpu 2 --num_label_raven 10000 &
nohup python train_drule_ws.py --gpu 3 --num_label_raven 10000 &
wait
nohup python train_drule_ws.py --gpu 0 --num_label_raven 5000 &
nohup python train_drule_ws.py --gpu 1 --num_label_raven 5000 &
nohup python train_drule_ws.py --gpu 2 --num_label_raven 5000 &
nohup python train_drule_ws.py --gpu 3 --num_label_raven 5000 &
wait
nohup python train_drule_ws.py --gpu 0 --num_label_raven 3000 &
nohup python train_drule_ws.py --gpu 1 --num_label_raven 3000 &
nohup python train_drule_ws.py --gpu 2 --num_label_raven 3000 &
nohup python train_drule_ws.py --gpu 3 --num_label_raven 3000 &
wait
nohup python train_drule_ws.py --gpu 0 --num_label_raven 1000 &
nohup python train_drule_ws.py --gpu 1 --num_label_raven 1000 &
nohup python train_drule_ws.py --gpu 2 --num_label_raven 1000 &
nohup python train_drule_ws.py --gpu 3 --num_label_raven 1000 &
