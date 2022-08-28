#!/bin/bash
#$ -N SAGraphTransformer_ZINC_b32-edge_16
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/16_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/SAGraphTransformer_molecules_ZINC_b32-edge.json --job_num 16 --pos_enc_dim 16 --log_file $fname
