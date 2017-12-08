#!/usr/bin/env bash


python train_cnn.py --embed=128 --hidden=25 --keep=0.3 --kernel_size=3 > results/cnn_test1.out &&
python train_cnn.py --embed=128 --hidden=50 --keep=0.3 --kernel_size=3 > results/cnn_test2.out &&
python train_cnn.py --embed=128 --hidden=100 --keep=0.3 --kernel_size=3 > results/cnn_test3.out &&
python train_cnn.py --embed=128 --hidden=25 --keep=0.4 --kernel_size=3 > results/cnn_test4.out &&
python train_cnn.py --embed=128 --hidden=50 --keep=0.4 --kernel_size=3 > results/cnn_test5.out &&
python train_cnn.py --embed=128 --hidden=100 --keep=0.4 --kernel_size=3 > results/cnn_test6.out &&
python train_cnn.py --embed=128 --hidden=25 --keep=0.5 --kernel_size=3 > results/cnn_test7.out &&
python train_cnn.py --embed=128 --hidden=50 --keep=0.5 --kernel_size=3 > results/cnn_test8.out &&
python train_cnn.py --embed=128 --hidden=100 --keep=0.5 --kernel_size=3 > results/cnn_test9.out &&

python train_cnn.py --embed=128 --hidden=25 --keep=0.3 --kernel_size=6 > results/cnn_test10.out &&
python train_cnn.py --embed=128 --hidden=50 --keep=0.3 --kernel_size=6 > results/cnn_test11.out &&
python train_cnn.py --embed=128 --hidden=100 --keep=0.3 --kernel_size=6 > results/cnn_test12.out &&
python train_cnn.py --embed=128 --hidden=25 --keep=0.4 --kernel_size=6 > results/cnn_test13.out &&
python train_cnn.py --embed=128 --hidden=50 --keep=0.4 --kernel_size=6 > results/cnn_test14.out &&
python train_cnn.py --embed=128 --hidden=100 --keep=0.4 --kernel_size=6 > results/cnn_test15.out &&
python train_cnn.py --embed=128 --hidden=25 --keep=0.5 --kernel_size=6 > results/cnn_test16.out &&
python train_cnn.py --embed=128 --hidden=50 --keep=0.5 --kernel_size=6 > results/cnn_test17.out &&
python train_cnn.py --embed=128 --hidden=100 --keep=0.5 --kernel_size=6 > results/cnn_test18.out &&

python train_cnn.py --embed=128 --hidden=25 --keep=0.3 --kernel_size=12 > results/cnn_test19.out &&
python train_cnn.py --embed=128 --hidden=50 --keep=0.3 --kernel_size=12 > results/cnn_test20.out &&
python train_cnn.py --embed=128 --hidden=100 --keep=0.3 --kernel_size=12 > results/cnn_test21.out &&
python train_cnn.py --embed=128 --hidden=25 --keep=0.4 --kernel_size=12 > results/cnn_test22.out &&
python train_cnn.py --embed=128 --hidden=50 --keep=0.4 --kernel_size=12 > results/cnn_test23.out &&
python train_cnn.py --embed=128 --hidden=100 --keep=0.4 --kernel_size=12 > results/cnn_test24.out &&
python train_cnn.py --embed=128 --hidden=25 --keep=0.5 --kernel_size=12 > results/cnn_test25.out &&
python train_cnn.py --embed=128 --hidden=50 --keep=0.5 --kernel_size=12 > results/cnn_test26.out &&
python train_cnn.py --embed=128 --hidden=100 --keep=0.5 --kernel_size=12 > results/cnn_test27.out

