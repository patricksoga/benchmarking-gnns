#!/bin/bash
#$ -N GraphTransformer_AQSOL_b128-bnorm-noedge-500k_20_2_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/20_DEBUG_2_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_AQSOL_b128-bnorm-noedge-500k.json --job_num 20 --pos_enc_dim 20 --log_file $fname --seed_array 41
