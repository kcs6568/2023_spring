#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

PORT=29500
START_GPU=3
NUM_GPU=4

cnt=0
while (( "${cnt}" < 2 )); do
    # ./run_three.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_AscendingSW_5e5_gatePosPG_DWA
    # ./run_three_retrain.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_AscendingSW_5e5_gatePosPG_DWA
    
    # ./run_three2.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_AscendingSW_5e5_gatePosPG_UWn2
    # ./run_three_retrain2.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_AscendingSW_5e5_gatePosPG_UWp2

    # ./run_three3.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_equalSW_5e5_gatePosPG_DWA
    # ./run_three_retrain3.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_equalSW_5e5_gatePosPG_DWA

    # ./run_three_retrain4.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW003_equalSW_5e5_gatePosPG_DWA_x2Train # no equal
    # ./run_three_retrain.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_AscendingSW_5e5_gatePosPG_DWA_x2Train
    # ./run_three_retrain2.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_AscendingSW_5e5_gatePosPG_UWp2_x2Train_GELU

    # ./run_three_retrain3.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_equalSW_5e5_gatePosPG_DWA
    # ./run_three_retrain.sh $PORT $START_GPU $NUM_GPU resnet50 gating pcgrad warm1K_SW002_AscendingSW_5e5_gatePosPG_DWA_x2Train


    ./run_three.sh $PORT $START_GPU $NUM_GPU resnet50 gating baseline SW002_AscendingSW_to5e5
    ./run_three_retrain.sh $PORT $START_GPU $NUM_GPU resnet50 gating baseline SW002_AscendingSW_to5e5
    ./run_three_retrain2.sh $PORT $START_GPU $NUM_GPU resnet50 gating baseline SW002_AscendingSW_to5e5_DWAp1
    



    (( cnt = "${cnt}" + 1 ))
done