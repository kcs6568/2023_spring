#!/bin/bash

PORT=$1
START_GPU=$2
NUM_GPU=$3
BACKBONE=$4
TASK=$5

TRAIN_ROOT=/root/src/gated_mtl/
CFG_PATH=/root/src/gated_mtl/cfgs/single_task/$4/$5.yaml
KILL_PROC="kill $(ps aux | grep single_train.py | grep -v grep | awk '{print $2}')"
TRAIN_FILE=single_train.py

TRAIN_SCRIPT=$TRAIN_ROOT/$TRAIN_FILE

DEVICES=""
d=$(($2-$3))
for ((n=$2; n>$d; n--))
do
    # $n > 0
    if [ $n -lt 0 ]; then 
        echo The gpu number $n is not valid. START_GPU: $2 / NUM_GPU: $3
        exit 1
    else
        DEVICES+=$n
        # $n < ~
        if [ $n -gt $(($d + 1)) ]
        then
            DEVICES+=,
        fi
    fi
done


# if [ $5 = resnet50 ]
# then
#     YAML_CFG=resnet50_clf_det_seg_1.yaml    

# elif [ $5 = resnext50 ]
# then
#     YAML_CFG=resnext50_32x4d_clf_det_seg_1.yaml

# elif [ $5 = mobilenetv3 ]
# then
#     YAML_CFG=mobile_v3_large_clf_det_seg_1.yaml
# else
#     echo Not supported backbone
# fi


SCH="multi"
OPT="sgd"
LR="1e-4"
GAMMA="0.1"
ADD_DISC="MTAN_detection"

# while : 
# do
for sch in $SCH
do
    # 1echo $sch
    for opt in $OPT
    do
        for lr in $LR
        do
            for gamma in $GAMMA
            do
                exp_case=nGPU"$4"_"$sch"_"$opt"_lr"$lr"
                
                if [ $sch != "cosine" ]
                then
                    exp_case="$exp_case"_gamma"$gamma"_$ADD_DISC
                else
                    exp_case="$exp_case"_$ADD_DISC
                fi

                CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$3 --master_port=$1 \
                    $TRAIN_SCRIPT --general \
                    --cfg $CFG_PATH \
                    --warmup-ratio 0.6 --workers 4 \
                    --exp-case "$exp_case" \
                    --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma
                    
                if [ $sch == "cosine" ]
                then
                    break
                fi
            done
        done
    done
done
# done

sleep 3