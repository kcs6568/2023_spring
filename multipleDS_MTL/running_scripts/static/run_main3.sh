#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

# ./run_single.sh 6000 3 4 resnet50 minicoco_mtan

# while :
# do
#     ./run_penta.sh 6000 3 4 resnet50 ucsmv
#     sleep 30
# done


# while :
# do
#     ./run_penta.sh 6000 3 4 resnet50 ucsmv
#     sleep 30
# done


# while :
# do
#     ./run_quad.sh 29500 mtan 3 4 resnet50
#     sleep 50

# done



# cd ../gating_scripts
# cd ./run_main2.sh



./run_three3.sh 6060 baseline 7 4 resnet50