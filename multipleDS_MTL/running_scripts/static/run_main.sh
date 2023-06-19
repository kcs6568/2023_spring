#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

cnt=0
while (( "${cnt}" < 1 )); do
    DATA_SEQ=c10_s10_mc_voc_ny
    ./run_seven.sh 29500 3 4 resnet50 static baseline NanTest_rankSeed_benchFalse $DATA_SEQ 1


    # ./run_single.sh 29500 3 4 resnet50 static baseline FrezBN_pret_Drop02 voc 1
    # ./run_single2.sh 29500 3 4 resnet50 static baseline pret stl10 1

    # ./run_single.sh 29500 3 4 resnet50 static baseline FrezBN_pret_Drop04 nyuv2 1
    # ./run_single.sh 29500 3 4 resnet50 static baseline nonFrezBN_pret_Drop04 nyuv2 2

    
    (( cnt = "${cnt}" + 1 ))
done


echo here
