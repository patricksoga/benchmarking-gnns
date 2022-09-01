#!/bin/bash
#$ -N GraphTransformer_ZINC_b128-bnorm-alt-noedge_178
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/178_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge.json --job_num 178 --pos_enc_dim 178 --log_file $fname
