#!/bin/bash
#$ -N GraphTransformer_SBM_PATTERN_b26-bnorm-alt-128-scale50-trials-scale002_128_0_95
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/128_DEBUG_0_95.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_PATTERN_b26-bnorm-alt-128-scale50-trials-scale002.json --job_num 128 --pos_enc_dim 128 --log_file $fname --seed_array 95
