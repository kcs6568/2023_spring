#!/bin/bash

cnt=0
while (( "${cnt}" < 2 )); do 
    ./run_three.sh 6000 3 4 resnet50 static_ddp pcgrad Basic_KaiNorm_totMean_allReduceStHe


    
    # ./run_three2.sh 29500 3 4 resnet50 static_ddp pcgrad Basic_KaiNorm_totMean_allReduce_UWneg1
    # ./run_three3.sh 29500 3 4 resnet50 static_ddp pcgrad Basic_KaiNorm_totMean_allReduce_UWpos2
    # ./run_three4.sh 29500 3 4 resnet50 static_ddp pcgrad Basic_KaiNorm_totMean_allReduce_DWA
    # ./run_three.sh 29500 3 4 resnet50 static_ddp pcgrad pnGrad_PNMG_minDIR_p2
    # ./run_three2.sh 29500 3 4 resnet50 static_ddp pcgrad pnGrad_PNtotalMG_minDIR_p2
    (( cnt = "${cnt}" + 1 ))
done


