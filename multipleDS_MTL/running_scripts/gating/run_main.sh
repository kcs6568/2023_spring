#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

PORT=6000
START_GPU=7
NUM_GPU=1

cnt=0
while (( "${cnt}" < 2 )); do
    ./run_three.sh $PORT $START_GPU $NUM_GPU resnet50 gating baseline SW002_AscendingSW
    # ./run_three_retrain.sh $PORT 7 1 resnet50 gating baseline SP002_clipG

    # ./run_three.sh 29500 3 4 resnet50 gating custom SP002_clipG
    # ./run_three_retrain.sh 29500 3 4 resnet50 gating custom SP002_clipG

    # ./run_three2.sh 29500 3 4 resnet50 gating custom SP0005_GELU
    # ./run_three_retrain2.sh 29500 3 4 resnet50 gating custom SP0005_GELU

    (( cnt = "${cnt}" + 1 ))
done