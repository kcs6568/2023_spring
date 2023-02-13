#!/bin/bash
PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5
TRAIN_ROOT=/root/src/gated_mtl/

KILL_PROC="kill $(ps aux | grep static_train.py | grep -v grep | awk '{print $2}')"
TRAIN_FILE=static_train.py
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

# echo $1 $2 $3 $4 $5 $6

# exit 1


# if [ $6 = csmvc ]
# then
#     CFG_PATH=/root/src/gated_mtl/cfgs/four_task/static/cifar10_stl10_minicoco_voc_city
# fi

if [ $5 = resnet50 ]
then
    YAML_CFG=resnet50_clf_det_seg_$2.yaml
fi

CFG_PATH=/root/src/gated_mtl/cfgs/three_task/static/cifar10_stl10_minicoco_voc/$YAML_CFG

SCH="cosine"
OPT="adamw"
LR="1e-4"
GAMMA="0.1"
ADD_DISC="noWarmup"

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

                CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$4 --master_port=$1 \
                    $TRAIN_SCRIPT --general \
                    --cfg $CFG_PATH \
                    --warmup-ratio -1 --workers 4 --grad-clip-value 1 \
                    --exp-case $exp_case --approach $2 \
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