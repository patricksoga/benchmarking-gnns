#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b32-bnorm-alt-study_32
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/32_DEBUG.log
touch $fname

cd ~/benchmarking-gnns/

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-bnorm-alt-study.json --job_num 32 --pos_enc_dim 32 --log_file $fname --gpu_id 4
