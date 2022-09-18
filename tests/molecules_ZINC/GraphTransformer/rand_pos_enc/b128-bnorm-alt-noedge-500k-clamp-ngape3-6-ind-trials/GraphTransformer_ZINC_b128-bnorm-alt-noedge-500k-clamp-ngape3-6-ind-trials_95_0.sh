#!/bin/bash
#$ -N GraphTransformer_ZINC_b128-bnorm-alt-noedge-500k-clamp-ngape3-6-ind-trials_6_0_95
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/6_DEBUG_0_95.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-500k-clamp-ngape3-6-ind-trials.json --job_num 6 --pos_enc_dim 6 --log_file $fname --seed_array 95
