nohup python train_mixmatch_rule.py --gpu 0 --num_label_raven 10000 &
nohup python train_mixmatch_rule.py --gpu 1 --num_label_raven 10000 &
nohup python train_mixmatch_rule.py --gpu 2 --num_label_raven 10000 &
nohup python train_mixmatch_rule.py --gpu 3 --num_label_raven 10000 &
wait
nohup python train_mixmatch_rule.py --gpu 0 --num_label_raven 20000 &
nohup python train_mixmatch_rule.py --gpu 1 --num_label_raven 20000 &
nohup python train_mixmatch_rule.py --gpu 2 --num_label_raven 20000 &
nohup python train_mixmatch_rule.py --gpu 3 --num_label_raven 20000 &
