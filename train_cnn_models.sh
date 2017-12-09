#!/usr/bin/env bash


python train_cnn.py --epochs=100 --embed=128 --hidden=25 --keep=0.3 --kernel_size=128 --model_name=cnn1 > results/cnn_test1.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=50 --keep=0.3 --kernel_size=128 --model_name=cnn2 > results/cnn_test2.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=100 --keep=0.3 --kernel_size=128 --model_name=cnn3 > results/cnn_test3.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=25 --keep=0.4 --kernel_size=128 --model_name=cnn4 > results/cnn_test4.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=50 --keep=0.4 --kernel_size=128 --model_name=cnn5 > results/cnn_test5.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=100 --keep=0.4 --kernel_size=128 --model_name=cnn6 > results/cnn_test6.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=25 --keep=0.5 --kernel_size=128 --model_name=cnn7 > results/cnn_test7.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=50 --keep=0.5 --kernel_size=128 --model_name=cnn8 > results/cnn_test8.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=100 --keep=0.5 --kernel_size=128 --model_name=cnn9 > results/cnn_test9.out

python train_cnn.py --epochs=100 --embed=128 --hidden=25 --keep=0.3 --kernel_size=56 --model_name=cnn10 > results/cnn_test1.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=50 --keep=0.3 --kernel_size=56 --model_name=cnn11 > results/cnn_test2.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=100 --keep=0.3 --kernel_size=56 --model_name=cnn12 > results/cnn_test3.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=25 --keep=0.4 --kernel_size=56 --model_name=cnn13 > results/cnn_test4.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=50 --keep=0.4 --kernel_size=56 --model_name=cnn14 > results/cnn_test5.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=100 --keep=0.4 --kernel_size=56 --model_name=cnn15 > results/cnn_test6.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=25 --keep=0.5 --kernel_size=56 --model_name=cnn16 > results/cnn_test7.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=50 --keep=0.5 --kernel_size=56 --model_name=cnn17 > results/cnn_test8.out &&
python train_cnn.py --epochs=100 --embed=128 --hidden=100 --keep=0.5 --kernel_size=56 --model_name=cnn18 > results/cnn_test9.out

