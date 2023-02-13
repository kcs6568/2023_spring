#!/bin/bash
PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5
TASK_SEQ=$6

TRAIN_ROOT=/root/src/gated_mtl/
CFG_PATH=/root/src/gated_mtl/cfgs/five_task/$2
KILL_PROC="kill $(ps aux | grep kcs_train2.py | grep -v grep | awk '{print $2}')"
TRAIN_FILE=kcs_train2.py
TRAIN_SCRIPT=$TRAIN_ROOT/$TRAIN_FILE
$KILL_PROC
# exit 1 


# make visible devices order automatically
DEVICES=""
d=$(($3-$4))
for ((n=$3; n>$d; n--))
do
    # $n > 0
    if [ $n -lt 0 ]; then 
        echo The gpu number $n is not valid. START_GPU: $3 / NUM_GPU: $4
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

if [ $6 = ucsmv ]
then
    CFG_PATH=/root/src/gated_mtl/cfgs/five_task/$2/usps_cifar10_stl10_minicoco_voc

elif [ $6 = mcsmv ]
then
    CFG_PATH=/root/src/gated_mtl/cfgs/five_task/$2/mnist_cifar10_stl10_minicoco_voc
fi

if [ $5 = resnet50 ]
then
    YAML_CFG=resnet50_clf_det_seg_2.yaml    
else
    echo Not supported backbone
fi


SCH="multi"
OPT="adamw"
LR="1e-4"
GAMMA="0.1"
ADD_DISC="002spW_general_G08temp5"

for sch in $SCH
do
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

                CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$4 --master_port=$1 \
                    $TRAIN_SCRIPT --general \
                    --cfg $CFG_PATH/$YAML_CFG \
                    --warmup-ratio -1 --workers 4 --grad-clip-value 1 \
                    --exp-case "$exp_case" \
                    --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma --resume
                    
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


