#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

cnt=0
while (( "${cnt}" < 2 )); do
    DATA_SEQ=c10_s10_mc_voc_ny

    # ./run_three.sh 29500 3 4 resnet50 static_ddp pcgrad nan_test

    ./run_seven.sh 29500 3 4 resnet50 static_ddp pcgrad DWA_encBias_PRelu $DATA_SEQ 1
    # ./run_seven2.sh 29500 3 4 resnet50 static_ddp pcgrad DWA_noDep $DATA_SEQ 2

    
    (( cnt = "${cnt}" + 1 ))
done

