#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b32-bnorm-alt-ngape3-8-ind-trials_8_0_95
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/8_DEBUG_0_95.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-bnorm-alt-ngape3-8-ind-trials.json --job_num 8 --pos_enc_dim 8 --log_file $fname --seed_array 95
