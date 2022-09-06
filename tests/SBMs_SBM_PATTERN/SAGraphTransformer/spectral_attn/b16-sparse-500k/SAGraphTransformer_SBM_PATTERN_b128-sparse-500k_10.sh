#!/bin/bash
#$ -N SAGraphTransformer_SBM_PATTERN_b128-sparse-500k_10
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/10_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/SAGraphTransformer_SBMs_SBM_PATTERN_b128-sparse-500k.json --job_num 10 --pos_enc_dim 10 --log_file $fname
