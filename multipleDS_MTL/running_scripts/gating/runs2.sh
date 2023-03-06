#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5



# while :
# do
#     ./run_main.sh
#     ./run_main2.sh
# done

./run_main2.sh
./run_three3.sh 6005 baseline 7 3 resnet50


