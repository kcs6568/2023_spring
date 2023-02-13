#!/bin/bash
PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5
TASK_SEQ=$6
START_CASE=$7
# START_CASE=${6:=""}

TRAIN_ROOT=/root/src/gated_mtl/
# CFG_PATH=/root/src/gated_mtl/cfgs/four_task/$2
KILL_PROC="kill $(ps aux | grep kcs_retrain2.py | grep -v grep | awk '{print $2}')"
TRAIN_FILE=kcs_retrain2.py
TRAIN_SCRIPT=$TRAIN_ROOT/$TRAIN_FILE
# $KILL_PROC
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
    CFG_ROOT=/root/src/gated_mtl/cfgs/five_task/$2/usps_cifar10_stl10_minicoco_voc
elif [ $6 = csmv ]
then
    CFG_ROOT=/root/src/gated_mtl/cfgs/four_task/$2/cifar10_stl10_minicoco_voc
elif [ $6 = mcsmv ]
then
    CFG_ROOT=/root/src/gated_mtl/cfgs/five_task/$2/mnist_cifar10_stl10_minicoco_voc
fi

if [ $5 = resnet50 ]
then
    YAML_NAME=resnet50_clf_det_seg_retrain_2.yaml    
else
    echo Not supported backbone
fi

CFG_PATH=$CFG_ROOT/$YAML_NAME


SCH="step"
OPT="adamw"
# LR="1e-5"
LR="2e-5"
GAMMA="0.8"
# ADD_DISC="Retrain_AllDynamic_fromStatic"
ADD_DISC="general_start8e["$7"]"

# while : 
# do
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
                    --cfg $CFG_PATH \
                    --warmup-ratio -1 --workers 4 --grad-clip-value 1 \
                    --exp-case "$exp_case" \
                    --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma --load-trained $7 --resume
                    

                if [ $sch == "cosine" ]
                then
                    break
                fi
            done

            # sleep 50
            # $KILL_PROC

        done
    done
done

