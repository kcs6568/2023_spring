#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5
TRAIN_ROOT=/root/src/gated_mtl/
CFG_PATH=/root/src/gated_mtl/cfgs/single_task/$5
KILL_PROC="kill $(ps aux | grep kcs_train1.py | grep -v grep | awk '{print $2}')"
TRAIN_FILE=kcs_train1.py
TRAIN_SCRIPT=$TRAIN_ROOT/$TRAIN_FILE

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

END

DO_CLS=0
DO_DET=1
DO_SEG=0

################################
##### CIFAR10 STL10 Process #####
################################

if [ $DO_CLS -eq 1 ]
then
    SCH="step"
    OPT="sgd adam"
    LR="0.001"
    GAMMA="0.1 0.25"
    ADD_DISC="nopret_freezeback"

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
                        --cfg $CFG_PATH/cifar10_no_transfer.yaml \
                        --exp-case "$exp_case" --workers 4 --warmup-ratio -1 \
                        --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma 
                       
                    
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


# if [ $DO_CLS -eq 1 ]
# then
#     SCH="cosine"
#     OPT="sgd"
#     LR="0.1"
#     GAMMA="0.1"
#     ADD_DISC="noclip_bs96"

#     for sch in $SCH
#     do
#         # echo $sch
#         for opt in $OPT
#         do
#             for lr in $LR
#             do
#                 for gamma in $GAMMA
#                 do
#                     $KILL_PROC
#                     exp_case=nGPU"$4"_"$sch"_"$opt"_lr"$lr"

#                     if [ $sch != "cosine" ]
#                     then
#                         exp_case="$exp_case"_gamma"$gamma"_$ADD_DISC
#                     else
#                         exp_case="$exp_case"_$ADD_DISC
#                     fi

#                     # CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$4 --master_port=$1 \
#                     #     $TRAIN_SCRIPT --general \
#                     #     --cfg $CFG_PATH/cifar10.yaml \
#                     #     --exp-case "$exp_case"_pretrained --warmup-ratio -1 \
#                     #     --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma \
                        
                    
#                     # $KILL_PROC
#                     # sleep 5

#                     CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$4 --master_port=$1 \
#                         $TRAIN_SCRIPT --general \
#                         --cfg $CFG_PATH/cifar10_no_transfer.yaml \
#                         --exp-case "$exp_case"_no_pretrained --warmup-ratio -1 \
#                         --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma \
                        
                    
#                     $KILL_PROC
#                     sleep 5

#                     if [ $sch == "cosine" ]
#                     then
#                         break
#                     fi

#                 done
#             done
#         done
#     done
# fi

###############################
####   MINICOCO Process   #####
###############################

if [ $DO_DET -eq 1 ]
then
    SCH="multi"
    OPT="sgd"
    LR="0.02"
    GAMMA="0.1"
    ADD_DISC="det_single_flops"

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
                        --task-bs 1 --get-flops

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
    SCH="lambda"
    OPT="sgd"
    LR="0.02"
    GAMMA="0.25 0.1"
    ADD_DISC="noclip_nowarmup_lrlambda0.9"

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
                        --exp-case $exp_case --warmup-ratio -1 --workers 8 \
                        --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma \
                        --task-bs 8

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

sleep 3
./terminate.sh $TRAIN_FILE