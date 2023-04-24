#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5


cnt=0
while (( "${cnt}" < 2 )); do
    ./run_three.sh 29500 3 4 resnet50 static pcgrad sepGrad_kaimNorm_Clip1_gradReduce
    # ./run_three2.sh 29500 3 4 resnet50 static pcgrad sepGrad_kaimNorm_eachClip1_UWneg1
    # ./run_three3.sh 29500 3 4 resnet50 static pcgrad sepGrad_kaimNorm_eachClip1_UWpos2
    # ./run_three4.sh 29500 3 4 resnet50 static pcgrad sepGrad_kaimNorm_eachClip1_DWA
    (( cnt = "${cnt}" + 1 ))
done

