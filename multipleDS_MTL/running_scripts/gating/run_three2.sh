#!/bin/bash
PORT=$1
START_GPU=$2
NUM_GPU=$3
BACKBONE=$4
METHOD=$5
APPROACH=$6
DESC=$7
TRAIN_ROOT=/root/2023_spring/multipleDS_MTL

KILL_PROC="kill $(ps aux | grep gating_train.py | grep -v grep | awk '{print $2}')"
TRAIN_FILE=gating_train.py
TRAIN_SCRIPT=$TRAIN_ROOT/$TRAIN_FILE
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


# if [ $6 = csmvc ]
# then
#     CFG_PATH=/root/src/gated_mtl/cfgs/four_task/static/cifar10_stl10_minicoco_voc_city
# fi

YAML_CFG=resnet50_2.yaml
CFG_PATH=$TRAIN_ROOT/cfgs/three_task/$5/$6/cifar10_minicoco_voc/$YAML_CFG

SCH="cosine"
OPT="adamw"
LR="1e-4"
GAMMA="0"
DESC_PART=""

if [ -z $7 ]
then
    ADD_DESC=$DESC_PART
else
    if [ -z $DESC_PART ]
    then
        ADD_DESC=$7
    else
        ADD_DESC=$DESC_PART"_"$7
    fi
fi


for sch in $SCH
do
    # echo $sch
    for opt in $OPT
    do
        for gamma in $GAMMA
        do
            for lr in $LR
            do
                exp_case=nGPU"$3"_"$sch"_"$opt"_lr"$lr"
                
                if [ $sch != "cosine" ]
                then
                    exp_case="$exp_case"_gamma"$gamma"_$ADD_DESC
                else
                    exp_case="$exp_case"_$ADD_DESC
                fi

                CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$3 --master_port=$1 \
                    $TRAIN_SCRIPT --general \
                    --cfg $CFG_PATH \
                    --warmup-ratio 1000 --workers 4 \
                    --exp-case $exp_case --grad-to-none \
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