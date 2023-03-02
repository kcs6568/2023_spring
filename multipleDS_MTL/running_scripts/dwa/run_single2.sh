#!/bin/bash

GPUS=$1
PORT=$2
METHOD=$3
NUM_GPU=$4
BACKBONE=$5
TRAIN_ROOT=/root/src/mtl_cl/
CFG_PATH=/root/src/mtl_cl/cfgs/single_task/$5
KILL_PROC="kill $(ps aux | grep train3.py | grep -v grep | awk '{print $2}')"

TRAIN_SCRIPT=$TRAIN_ROOT/train3.py

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

: <<'END'
<CIFAR10-STL10-minicoco-voc>
 - SGD
    
        
<CIFAR100-minicoco-voc>
 - SGD

### ONGOING ###



### To Do ###

END

if [ $5 = resnet50 ]
then
    YAML_CFG=resnet50_clf_det_seg.yaml    

elif [ $5 = mobilenetv3 ]
then
    YAML_CFG=mobile_v3_large_clf_det_seg.yaml
else
    echo Not supported backbone
fi

DO_CLS=1
DO_DET=0
DO_SEG=0

################################
##### CIFAR10 STL10 Process #####
################################


if [ $DO_CLS -eq 1 ]
then
    SCH="cosine"
    OPT="adam"
    LR="0.1"
    GAMMA="0.1"
    ADD_DISC="noclip_bs64_adam_eps1e-4"

    for sch in $SCH
    do
        # echo $sch
        for opt in $OPT
        do
            for lr in $LR
            do
                for gamma in $GAMMA
                do
                    $KILL_PROC
                    exp_case=nGPU"$4"_"$sch"_"$opt"_lr"$lr"

                    if [ $sch != "cosine" ]
                    then
                        exp_case="$exp_case"_gamma"$gamma"_$ADD_DISC
                    else
                        exp_case="$exp_case"_$ADD_DISC
                    fi

                    # CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$4 --master_port=$1 \
                    #     $TRAIN_SCRIPT --general \
                    #     --cfg $CFG_PATH/cifar10.yaml \
                    #     --exp-case "$exp_case"_pretrained --warmup-ratio -1 \
                    #     --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma \
                        
                    
                    # $KILL_PROC
                    # sleep 5

                    CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$4 --master_port=$1 \
                        $TRAIN_SCRIPT --general \
                        --cfg $CFG_PATH/cifar10_no_transfer.yaml \
                        --exp-case "$exp_case"_no_pretrained --warmup-ratio -1 \
                        --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma \
                        
                    
                    $KILL_PROC
                    sleep 5

                    if [ $sch == "cosine" ]
                    then
                        break
                    fi

                done
            done
        done
    done
fi


###############################
####   MINICOCO Process   #####
###############################

if [ $DO_DET -eq 1 ]
then
    SCH="multi"
    OPT="sgd"
    LR="0.02"
    GAMMA="0.1"
    ADD_DISC="clip1"

    for sch in $SCH
    do
        # echo $sch
        for opt in $OPT
        do
            for lr in $LR
            do
                for gamma in $GAMMA
                do
                    $KILL_PROC
                    $KILL_PROC
                    $KILL_PROC
                    $KILL_PROC
                    $KILL_PROC

                    exp_case=nGPU"$4"_"$sch"_"$opt"_lr"$lr"

                    if [ $sch != "cosine" ]
                    then
                        exp_case="$exp_case"_gamma"$gamma"_$ADD_DISC
                    else
                        exp_case="$exp_case"_$ADD_DISC
                    fi


                    CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$4 --master_port=$1 \
                        $TRAIN_SCRIPT --general \
                        --cfg $CFG_PATH/minicoco.yaml \
                        --exp-case $exp_case --warmup-ratio -1 --workers 8 --grad-clip-value 1 \
                        --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma \
                        --task-bs 2 --resume 

                    $KILL_PROC
                    $KILL_PROC
                    $KILL_PROC
                    $KILL_PROC
                    $KILL_PROC
                    sleep 5
                    
                    if [ $sch == "cosine" ]
                    then
                        break
                    fi
                done
            done
        done
    done
fi


################################
#####    VOC0712 Process   #####
################################

if [ $DO_SEG -eq 1 ]
then
    SCH="multi cosine"
    OPT="sgd"
    LR="0.01 0.15"
    GAMMA="0.25 0.1"
    ADD_DISC="clip1"

    for sch in $SCH
    do
        # echo $sch
        for opt in $OPT
        do
            for lr in $LR
            do
                for gamma in $GAMMA
                do
                    $KILL_PROC
                    exp_case=nGPU"$4"_"$sch"_"$opt"_lr"$lr"

                    if [ $sch != "cosine" ]
                    then
                        exp_case="$exp_case"_gamma"$gamma"_$ADD_DISC
                    else
                        exp_case="$exp_case"_$ADD_DISC
                    fi


                    CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$4 --master_port=$1 \
                        $TRAIN_SCRIPT --general \
                        --cfg $CFG_PATH/voc0712.yaml \
                        --exp-case $exp_case --warmup-ratio -1 --workers 8 --grad-clip-value 1 \
                        --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma \
                        --task-bs 2

                    $KILL_PROC
                    sleep 5

                    if [ $sch == "cosine" ]
                    then
                        break
                    fi

                done
            done
        done
    done
fi
