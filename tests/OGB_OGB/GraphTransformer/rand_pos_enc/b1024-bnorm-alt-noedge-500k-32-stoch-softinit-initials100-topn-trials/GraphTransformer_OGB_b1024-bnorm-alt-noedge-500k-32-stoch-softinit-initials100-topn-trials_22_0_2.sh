#!/bin/bash
#$ -N GraphTransformer_OGB_b1024-bnorm-alt-noedge-500k-32-stoch-softinit-initials100-topn-trials_32_0_22
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/32_DEBUG_0_22.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_OGB_graph_regression.py --config tests/test-configs/GraphTransformer_OGB_OGB_b1024-bnorm-alt-noedge-500k-32-stoch-softinit-initials100-topn-trials.json --job_num 32 --pos_enc_dim 32 --log_file $fname --seed_array 22
