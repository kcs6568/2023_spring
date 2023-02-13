#!/bin/bash

# PORT=$1
# METHOD=$2
# START_GPU=$3
# NUM_GPU=$4
# BACKBONE=$5

while :
do
    ./run_penta2.sh 6000 baseline 3 4 resnet50 ucsmv
    sleep 50

    ./run_retrain2.sh 6000 baseline 3 4 resnet50 ucsmv nGPU4_multi_adamw_lr1e-4_gamma0.1_002spW_general_G08temp5
    sleep 50

done


