#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

while :
do
    ./run_three2.sh 6000 baseline 7 2 resnet50
done
