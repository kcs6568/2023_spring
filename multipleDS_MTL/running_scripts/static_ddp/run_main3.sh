#!/bin/bash

cnt=0
while (( "${cnt}" < 1 )); do
    ./run_two2.sh 29500 3 2 resnet50 static_ddp pcgrad mini_voc_test
    (( cnt = "${cnt}" + 1 ))
done