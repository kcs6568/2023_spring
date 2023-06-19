#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5
PORT=6000


cnt=0
while (( "${cnt}" < 2 )); do
    # DATA_SEQ=c10_s10_mc_voc_ny

    ./run_single3.sh $PORT 3 4 resnet50 static baseline pret voc 1

    # ./run_single.sh $PORT 3 4 resnet50 static baseline FrezBN_Dep_pret_Drop04 nyuv2 3
    # ./run_single.sh $PORT 3 4 resnet50 static baseline nonFrezBN_Dep_pret_Drop04 nyuv2 4

    # ./run_single.sh $PORT 3 4 resnet50 static baseline FrezBN_SN_pret_Drop04 nyuv2 5
    # ./run_single.sh $PORT 3 4 resnet50 static baseline nonFrezBN_SN_pret_Drop04 nyuv2 6

    # ./run_single2.sh $PORT 3 4 resnet50 static baseline encBias_bs4_scrat200E_SSeg_BN nyuv2 1
    # ./run_single.sh $PORT 3 4 resnet50 static baseline encBias_bs4_scrat200E_SN_warm12_BN nyuv2 2
    # ./run_single.sh $PORT 3 4 resnet50 static baseline encBias_bs4_scrat200E_Depth_warm12_BN nyuv2 3


    # ./run_single.sh $PORT 3 4 resnet50 static baseline allBias_SSeg_warm12_BN_pAcc nyuv2 1
    # ./run_single.sh $PORT 3 4 resnet50 static baseline allBias_SN_warm12_BN nyuv2 2
    # ./run_single.sh $PORT 3 4 resnet50 static baseline allBias_Depth_warm12_BN nyuv2 3
    
    (( cnt = "${cnt}" + 1 ))
done


echo here
