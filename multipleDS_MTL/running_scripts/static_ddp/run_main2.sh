#!/bin/bash

cnt=0
while (( "${cnt}" < 2 )); do
    ./run_three.sh 6000 7 4 resnet50 static_ddp pcgrad Basic_KaiNorm_totMean_allReduce
    # ./run_three3.sh 6000 3 4 resnet50 static_ddp pcgrad basicPCGrad_UWpos2_clipGrad
    # ./run_three3.sh 6000 3 4 resnet50 static_ddp pcgrad DDPHPS_basicMean2
    (( cnt = "${cnt}" + 1 ))
done