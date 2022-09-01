#!/bin/bash
#$ -N GraphTransformer_ZINC_b128-bnorm-alt-noedge_212
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/212_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge.json --job_num 212 --pos_enc_dim 212 --log_file $fname
