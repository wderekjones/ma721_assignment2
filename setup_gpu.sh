#!/usr/bin/env bash

LD_LIBRARY_PATH+=:/u/amo-d0/lib/cuda-8.0_cudnn-v6/lib64
LD_LIBRARY_PATH+=:$ANACONDA_HOME/lib

export LD_LIBRARY_PATH

source activate torch3
