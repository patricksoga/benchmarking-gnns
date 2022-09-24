#!/bin/bash
#$ -N SAGraphTransformer_SBM_CLUSTER_b32-sparse-500k-lr35-trials_10_0_95
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/10_DEBUG_0_95.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/SAGraphTransformer_SBMs_SBM_CLUSTER_b32-sparse-500k-lr35-trials.json --job_num 10 --pos_enc_dim 10 --log_file $fname --seed_array 95
