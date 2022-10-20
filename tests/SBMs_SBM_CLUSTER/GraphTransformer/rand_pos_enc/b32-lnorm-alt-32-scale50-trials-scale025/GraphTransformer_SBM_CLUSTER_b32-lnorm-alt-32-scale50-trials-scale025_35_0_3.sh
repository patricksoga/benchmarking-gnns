#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b32-lnorm-alt-32-scale50-trials-scale025_32_0_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/32_DEBUG_0_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-lnorm-alt-32-scale50-trials-scale025.json --job_num 32 --pos_enc_dim 32 --log_file $fname --seed_array 35
