#!/bin/bash
#$ -N SAGraphTransformer_ZINC_b128-noedge-500k_8
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/8_DEBUG_3lpe.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/SAGraphTransformer_molecules_ZINC_b128-noedge-500k.json --job_num 8 --pos_enc_dim 8 --log_file $fname --lpe_layers 3
