#!/bin/bash

# PORT=$1
# METHOD=$2
# START_GPU=$3
# NUM_GPU=$4
# BACKBONE=$5

# while :
# do
#     ./run_quad.sh 6005 baseline 3 4 resnet50 
#     # sleep 50
#     # ./run_quad2.sh 6006 baseline 3 4 resnet50 
#     # sleep 50
# done


# while :
# do
#     ./run_retrain.sh 6005 baseline 3 4 resnet50 nGPU4_multi_adamw_lr1e-4_gamma0.1_blockSW_001sp_G08temp5
#     sleep 50
    
#     ./run_retrain2.sh 6005 baseline 3 4 resnet50 nGPU4_multi_adamw_lr1e-4_gamma0.1_blockSW_001sp_G08temp5
#     sleep 50
# done


# ./run_retrain.sh 6005 baseline 3 4 resnet50 nGPU4_multi_adamw_lr1e-4_gamma0.1_OptimalTest_noshL_0015sp_G08temp5
# ./run_retrain2.sh 6005 baseline 3 4 resnet50 nGPU4_multi_adamw_lr1e-4_gamma0.1_OptimalTest_noshL_0015sp_G08temp5

# while :
# do
#     ./run_penta.sh 6005 baseline 3 4 resnet50 ucsmv

#     sleep 50
# done

# while :
# do
#     # ./run_penta.sh 6000 baseline 3 4 resnet50 ucsmv
#     # sleep 50uy

#     # ./run_penta2.sh 6000 baseline 3 4 resnet50 ucsmv
#     # sleep 50

#     # ./run_retrain.sh 6005 baseline 3 4 resnet50 nGPU4_multi_adamw_lr1e-4_gamma0.1_OptimalTest_001spW_generalRatio_G08temp5
#     # ./run_retrain2.sh 6005 baseline 3 4 resnet50 nGPU4_multi_adamw_lr1e-4_gamma0.1_OptimalTest_noshL_0015sp_G08temp5
    
# done

# ./run_retrain.sh 6005 baseline 3 4 resnet50 ucsmv nGPU4_multi_adamw_lr1e-4_gamma0.1_OptimalTest_001spW_generalRatio_G08temp5


# while :
# do
#     # ./run_quad.sh 6005 baseline 3 4 resnet50 
#     # sleep 50

#     ./run_penta.sh 6005 baseline 3 4 resnet50 mcsmv
#     sleep 50

#     ./run_retrain.sh 6005 baseline 3 4 resnet50 mcsmv nGPU4_multi_adamw_lr1e-4_gamma0.1_Dynamic_0015spW_general_start8Eto12E_G08temp5
#     sleep 50

#     ./run_penta2.sh 6005 baseline 3 4 resnet50 mcsmv
#     sleep 50

#     ./run_retrain2.sh 6005 baseline 3 4 resnet50 mcsmv nGPU4_multi_adamw_lr1e-4_gamma0.1_Dynamic_002spW_general_start8Eto12E_G08temp5
#     sleep 50

#     # ./run_retrain2.sh 6005 baseline 3 4 resnet50 ucsmv nGPU4_multi_adamw_lr1e-4_gamma0.1_Dynamic_0001spW_general_taskSparsity12133_G08temp5
#     # sleep 50

#     # ./run_penta.sh 6005 baseline 3 4 resnet50 ucsmv
#     # sleep 50

#     # ./run_retrain.sh 6005 baseline 3 4 resnet50 ucsmv nGPU4_multi_adamw_lr1e-4_gamma0.1_Dynamic_001spW_general_taskSparsity12133_G08temp5
#     # sleep 50
# done

# ./run_penta.sh 6005 baseline 3 4 resnet50 mcsmv
# sleep 50




# mnist process
# ./run_retrain.sh 6005 baseline 3 4 resnet50 mcsmv nGPU4_multi_adamw_lr1e-4_gamma0.1_Dynamic_0015spW_general_start8Eto12E_G08temp5
# sleep 50

# ./run_penta2.sh 6005 baseline 3 4 resnet50 mcsmv
# sleep 50

# ./run_retrain2.sh 6005 baseline 3 4 resnet50 mcsmv nGPU4_multi_adamw_lr1e-4_gamma0.1_Dynamic_002spW_general_start8Eto12E_G08temp5
# sleep 50



# ./run_retrain.sh 6005 baseline 7 4 resnet50 csmv nGPU4_multi_adamw_lr1e-4_gamma0.1_blockSW_001sp_G08temp5

while :
do
    # ./run_quad.sh 6005 baseline 7 4 resnet50 
    # sleep 50

    # ./run_quad.sh 6005 disparse 7 4 resnet50 csmv
    # sleep 50

    # ./run_retrain.sh 6005 baseline 7 4 resnet50 csmv nGPU4_multi_adamw_lr1e-4_gamma0.1_Dynaimic_general_009spL_toEpoch8_G08temp5
    # sleep 50

    # ./run_quad2.sh 6005 baseline 7 4 resnet50 
    # sleep 50

    # ./run_penta.sh 6005 baseline 3 4 resnet50 ucsmv
    # sleep 50

    ./run_retrain.sh 6005 baseline 7 1 resnet50 csmv nGPU4_multi_adamw_lr1e-4_gamma0.1_Dynaimic_general_009spL_toEpoch8_G08temp5
    sleep 50


done
