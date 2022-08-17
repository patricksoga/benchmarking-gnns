#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b2-bnorm-alt
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/16_DEBUG.log
touch $fname
cd; cd benchmarking-gnns;

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b2-bnorm-alt.json --job_num 16 --pos_enc_dim 16 --log_file $fname --batch_size 32
