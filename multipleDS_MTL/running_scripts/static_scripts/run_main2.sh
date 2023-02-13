#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

# while :
# do
#     ./run_penta.sh 6000 3 4 resnet50 ucsmv
#     sleep 30
# done
# while :
# do
#     ./run_penta.sh 6000 3 4 resnet50 mcsmv
#     sleep 30
# done

while :
do
    ./run_quad2.sh 6000 mtan 7 4 resnet50
    sleep 50

done



# cd ../gating_scripts
# cd ./run_main2.sh
