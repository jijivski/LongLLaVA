#!/bin/bash

# conda activate LongLLaVa

########################## Setup model_id and checkpoint_dir ##########################

# MODLE_DIR="./ckpts/10SFT2dSenseLong176K" # directory of the checkpoint
# experiment_name="test_eval_241030"

MODLE_DIR="./ckpts/MultiImageSFT"
experiment_name="sft_eval_241107"

patchStrategy="norm"   # [bestFit, norm]
T=1.0
FrameNum=128
# MODEL_ID="orig_10SFT2dSenseLong176K_241101" # a unique ID for the checkpoint that can be used to retrieve results
MODEL_ID="sft_10SFT2dSenseLong176K_241108" # a unique ID for the checkpoint that can be used to retrieve results

log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
########################## Setup model_id and checkpoint_dir ##########################

PATH_TO_ALL_RESULTS="./benchmark_results/$MODEL_ID.txt"


# ########################## Run each benchmark sequentially ##########################
# GQA
# bash benchmarks/GQA/eval_gqa.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy > ${log_folder}/GQA.log 2>&1

# # MM-Vet
# # gpt-4-0613
# bash benchmarks/MM-Vet/eval_mmvet.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy > ${log_folder}/MM-Vet.log 2>&1

# # MMBench-en
# bash benchmarks/MMBench/eval_mmbench_all_in_one.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy > ${log_folder}/MMBench-en.log 2>&1

# # MME
# bash benchmarks/MME/eval_mme.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy > ${log_folder}/MME.log 2>&1

# # MMMU
# bash benchmarks/MMMU/eval_mmmu.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy > ${log_folder}/MMMU.log 2>&1

# # POPE
# bash benchmarks/POPE/eval_pope.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy > ${log_folder}/POPE.log 2>&1

# # ScienceQA
# bash benchmarks/ScienceQA/eval_sqa.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy > ${log_folder}/ScienceQA.log 2>&1

# # SEEDBench
# bash benchmarks/SEEDBench/eval_seedbench.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy > ${log_folder}/SEEDBench.log 2>&1

# # MileBench
# bash benchmarks/MileBench/scripts/eval_milebench.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS > ${log_folder}/milebench.log 2>&1

# VideoMME
# bash benchmarks/VideoMME/eval_videomme.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy $T $FrameNum > ${log_folder}/videomme.log 2>&1


# vstarbench
# test 100 passed
patchside_length=336
bash benchmarks/vstarbench/eval_vstarbench.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy $patchside_length> ${log_folder}/vstarbench_norm.log 2>&1
# bash benchmarks/vstarbench/eval_vstarbench.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy $patchside_length

# VIAH
# test 100 passed
bash benchmarks/VIAH/eval_viah.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy $T $FrameNum > ${log_folder}/viah.log 2>&1
# bash benchmarks/VIAH/eval_viah.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $patchStrategy $T $FrameNum 


# ########################## Run each benchmark sequentially ##########################
