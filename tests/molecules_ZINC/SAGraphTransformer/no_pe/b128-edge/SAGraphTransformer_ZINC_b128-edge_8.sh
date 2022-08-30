#!/bin/bash
#$ -N SAGraphTransformer_ZINC_b128-edge_8
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/8_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/SAGraphTransformer_molecules_ZINC_b128-edge.json --job_num 8 --pos_enc_dim 8 --log_file $fname
