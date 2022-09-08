#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b32-bnorm-alt-study_2
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/2_DEBUG.log
touch $fname

cd ~/benchmarking-gnns/

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-bnorm-alt-study.json --job_num 2 --pos_enc_dim 2 --log_file $fname --gpu_id 0
