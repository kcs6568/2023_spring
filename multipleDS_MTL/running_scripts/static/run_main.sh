#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5


cnt=0
while (( "${cnt}" < 2 )); do
    ./run_three.sh 29500 3 4 resnet50 static baseline w1000_FFT_leakyRelu_2
    (( cnt = "${cnt}" + 1 ))
done

