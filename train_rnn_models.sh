#!/usr/bin/env bash


python train_rnn.py --epochs=20 --embed=128 --hidden=25 --keep=0.3 --model_name=rnn1 > results/rnn_test1.out &&
python train_rnn.py --epochs=20 --embed=128 --hidden=50 --keep=0.3 --model_name=rnn2 > results/rnn_test2.out &&
python train_rnn.py --epochs=20 --embed=128 --hidden=100 --keep=0.3 --model_name=rnn3 > results/rnn_test3.out &&
python train_rnn.py --epochs=20 --embed=128 --hidden=25 --keep=0.4 --model_name=rnn4 > results/rnn_test4.out &&
python train_rnn.py --epochs=20 --embed=128 --hidden=50 --keep=0.4 --model_name=rnn5 > results/rnn_test5.out &&
python train_rnn.py --epochs=20 --embed=128 --hidden=100 --keep=0.4 --model_name=rnn6 > results/rnn_test6.out &&
python train_rnn.py --epochs=20 --embed=128 --hidden=25 --keep=0.5 --model_name=rnn7 > results/rnn_test7.out &&
python train_rnn.py --epochs=20 --embed=128 --hidden=50 --keep=0.5 --model_name=rnn8 > results/rnn_test8.out &&
python train_rnn.py --epochs=20 --embed=128 --hidden=100 --keep=0.5 --model_name=rnn9 > results/rnn_test9.out
