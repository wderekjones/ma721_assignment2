#!/usr/bin/env bash


python train_rnn.py --embed=128 --hidden=25 --keep=0.3 > results/rnn_test1.out &&
python train_rnn.py --embed=128 --hidden=50 --keep=0.3 > results/rnn_test2.out &&
python train_rnn.py --embed=128 --hidden=100 --keep=0.3 > results/rnn_test3.out &&
python train_rnn.py --embed=128 --hidden=25 --keep=0.4 > results/rnn_test4.out &&
python train_rnn.py --embed=128 --hidden=50 --keep=0.4 > results/rnn_test5.out &&
python train_rnn.py --embed=128 --hidden=100 --keep=0.4> results/rnn_test6.out &&
python train_rnn.py --embed=128 --hidden=25 --keep=0.5 > results/rnn_test7.out &&
python train_rnn.py --embed=128 --hidden=50 --keep=0.5 > results/rnn_test8.out &&
python train_rnn.py --embed=128 --hidden=100 --keep=0.5 > results/rnn_test9.out
