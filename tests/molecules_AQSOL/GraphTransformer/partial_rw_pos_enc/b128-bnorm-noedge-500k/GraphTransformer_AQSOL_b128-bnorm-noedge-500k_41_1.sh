#!/bin/bash
#$ -N GraphTransformer_AQSOL_b128-bnorm-noedge-500k_16_1_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/16_DEBUG_1_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_AQSOL_b128-bnorm-noedge-500k.json --job_num 16 --pos_enc_dim 16 --log_file $fname --seed_array 41
