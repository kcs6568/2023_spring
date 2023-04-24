#!/bin/bash

cnt=0
while (( "${cnt}" < 2 )); do
    ./run_three2.sh 6066 5 2 resnet50 gating_ddp gradvac
    ./run_three2_retrain.sh 6066 5 2 resnet50 gating_ddp gradvac [From_gating_ddp]
    (( cnt = "${cnt}" + 1 ))
done