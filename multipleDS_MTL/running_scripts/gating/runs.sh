#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5



while :
do
    ./run_main.sh
    ./run_main2.sh
done

