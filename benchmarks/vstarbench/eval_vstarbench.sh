# #!/bin/bash

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

#!/bin/bash

CKPT=$1
mp=$2
path_to_all_results=$3
patchStrategy=$4
patchside_length=$5
name=${CKPT}_${patchStrategy}_${patchside_length}

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"
# gpu_list="6,7"
# gpu_list=""



read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

# CHUNKS=$(( (${#GPULIST[@]} + 1) / 2 ))
CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/vstarbench/model_vstar.py \
    --model-path $mp \
    --gt_file ./benchmarks/vstarbench/vstartest.json \
    --output_dir ./benchmarks/vstarbench/outputs/$name \
    --output_name pred \
    --patchStrategy $patchStrategy \
    --patchside_length $patchside_length \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &

    echo "CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/vstarbench/model_vstar.py "

done

wait

echo "python ./benchmarks/vstarbench/generate_score.py \
--output_path ./benchmarks/vstarbench/outputs/$name \
--score_path ./benchmarks/vstarbench/outputs/$name/score.json \
"


python ./benchmarks/vstarbench/generate_score.py \
    --output_path ./benchmarks/vstarbench/outputs/$name \
    --score_path ./benchmarks/vstarbench/outputs/$name/score.json \
