#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

./run_three2.sh 6000 baseline 3 2 resnet50
