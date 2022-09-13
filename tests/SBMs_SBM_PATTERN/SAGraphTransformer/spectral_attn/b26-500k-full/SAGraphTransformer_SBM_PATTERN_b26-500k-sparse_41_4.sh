#!/bin/bash
#$ -N SAGraphTransformer_SBM_PATTERN_b26-500k-sparse_32_4_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/32_DEBUG_4_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/SAGraphTransformer_SBMs_SBM_PATTERN_b26-500k-full.json --job_num 32 --pos_enc_dim 32 --log_file $fname --seed_array 41 --L 6 --n_heads 8 --hidden_dim 56 --out_dim 56
