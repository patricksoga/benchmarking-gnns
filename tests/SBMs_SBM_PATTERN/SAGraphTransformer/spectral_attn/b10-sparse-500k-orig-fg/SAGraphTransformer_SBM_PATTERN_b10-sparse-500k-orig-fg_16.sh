#!/bin/bash
#$ -N SAGraphTransformer_SBM_PATTERN_b10-sparse-500k-orig-fg_16
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/16_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/SAGraphTransformer_SBMs_SBM_PATTERN_b10-sparse-500k-orig-fg.json --job_num 16 --pos_enc_dim 16 --log_file $fname
