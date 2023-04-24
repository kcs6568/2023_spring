#!/bin/bash

cnt=0
while (( "${cnt}" < 2 )); do
    # ./run_three3.sh 6006 7 2 resnet50 gating_ddp cagrad
    ./run_three3_retrain.sh 6006 7 2 resnet50 gating_ddp cagrad [From_gating_ddp]
    (( cnt = "${cnt}" + 1 ))
done