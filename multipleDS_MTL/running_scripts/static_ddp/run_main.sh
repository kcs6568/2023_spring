#!/bin/bash

cnt=0
while (( "${cnt}" < 2 )); do 
    ./run_three.sh 29500 3 4 resnet50 static_ddp pcgrad warm1000
    ./run_three2.sh 29500 3 4 resnet50 static_ddp pcgrad warm1000_DWA

    
    (( cnt = "${cnt}" + 1 ))
done


