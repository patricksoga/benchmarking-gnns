#!/bin/bash
#$ -N SAGraphTransformer_ZINC_b128-noedge-500k-fg-trials_10_0_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/10_DEBUG_0_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/SAGraphTransformer_molecules_ZINC_b128-noedge-500k-fg-trials.json --job_num 10 --pos_enc_dim 10 --log_file $fname --seed_array 35
