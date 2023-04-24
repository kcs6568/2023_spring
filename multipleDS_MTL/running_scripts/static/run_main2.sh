#!/bin/bash

cnt=0
while (( "${cnt}" < 2 )); do
    ./run_three2.sh 6000 3 2 resnet50 static pcgrad sepGrad_interval1
    (( cnt = "${cnt}" + 1 ))
done

