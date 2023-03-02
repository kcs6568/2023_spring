#!/bin/bash

: <<'END'

[DET-SEG]
- minicoco-voc
    # 데이터셋의 크기, class의 수의 balance 정도가 크게 영향을 미치는 듯
    - cosine / 0.01 / default: 19.5 | 64.94
    - cosine / 0.02 / default: 22.33 | 64.1
    - multi / 0.01 / default: 20.24 | 55.63
    - multi / 0.02 / default: 
END
GPUS=$1
PORT=$2
# TASK_CFG=$3
CFG_PATH=/root/mtl_ci/cfgs/two_task
KILL_PROC="kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')"
for ((n=7; n>=((8-$GPUS)); n--))
do
    DEVICES+=$n

    if [ $n -gt $((8-$GPUS)) ]
    then
        DEVICES+=,
    fi
done

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     train.py \
#     --cfg $CFG_PATH/det_seg/minicoco_voc/resnet50_det_seg.yaml \
#     --exp-case deadlock_test \
#     --lr 0.01 --use-minids --bottleneck-test \

$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
    train.py \
    --cfg $CFG_PATH/clf_det/cifar10_minicoco/fasterrcnn_resnet50_clf_det.yaml \
    --exp-case deadlock_test \
    --lr 0.01 --use-minids --bottleneck-test \

$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     train.py \
#     --cfg $CFG_PATH/clf_seg/cifar10_voc/resnet50_clf_seg.yaml \
#     --exp-case deadlock_test \
#     --lr 0.01 --use-minids --bottleneck-test \

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

    