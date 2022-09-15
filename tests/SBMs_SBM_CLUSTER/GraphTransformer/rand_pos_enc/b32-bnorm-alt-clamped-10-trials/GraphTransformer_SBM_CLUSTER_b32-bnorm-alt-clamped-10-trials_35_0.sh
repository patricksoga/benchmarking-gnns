#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b32-bnorm-alt-clamped-10-trials_10_0_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/10_DEBUG_0_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-bnorm-alt-clamped-10-trials.json --job_num 10 --pos_enc_dim 10 --log_file $fname --seed_array 35
