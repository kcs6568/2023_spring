#!/bin/bash
PORT=$1
START_GPU=$2
NUM_GPU=$3
BACKBONE=$4
TASK_SEQ=$5
TRAIN_ROOT=/root/src/gated_mtl/

KILL_PROC="kill $(ps aux | grep static_train.py | grep -v grep | awk '{print static}')"
TRAIN_FILE=static_train.py
TRAIN_SCRIPT=$TRAIN_ROOT/$TRAIN_FILE
# $KILL_PROC
# exit 1 

# $KILL_PROC
# exit 1

# make visible devices order automatically
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

# echo $1 $2 $3 $4 $5 $6

# exit 1


if [ $5 = ucsmv ]
then
    CFG_PATH=/root/src/gated_mtl/cfgs/five_task/static/usps_cifar10_stl10_minicoco_voc
elif [ $5 = mcsmv ]
then
    CFG_PATH=/root/src/gated_mtl/cfgs/five_task/static/mnist_cifar10_stl10_minicoco_voc
fi

if [ $4 = resnet50 ]
then
    YAML_CFG=resnet50_clf_det_seg_1.yaml
fi

SCH="multi"
OPT="adamw"
LR="1e-4"
GAMMA="0.1"
ADD_DISC="general_8eMNIST"

for sch in $SCH
do
    # echo $sch
    for opt in $OPT
    do
        for gamma in $GAMMA
        do
            for lr in $LR
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
                    --cfg $CFG_PATH/$YAML_CFG \
                    --warmup-ratio -1 --workers 4 --grad-clip-value 1 \
                    --exp-case $exp_case \
                    --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma --resume

                sleep 5

                if [ $sch == "cosine" ]
                then
                    break
                fi


            done
        done
    done
done


sleep 3