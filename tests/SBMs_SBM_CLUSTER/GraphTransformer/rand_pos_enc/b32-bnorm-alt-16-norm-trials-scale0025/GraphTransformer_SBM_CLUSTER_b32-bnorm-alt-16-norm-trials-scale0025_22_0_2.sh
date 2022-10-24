#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b32-bnorm-alt-16-norm-trials-scale0025_16_0_22
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/16_DEBUG_0_22.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-bnorm-alt-16-norm-trials-scale0025.json --job_num 16 --pos_enc_dim 16 --log_file $fname --seed_array 22
