#!/bin/bash
#$ -N GraphTransformer_ZINC_b128-bnorm-alt-noedge_136
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/136_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge.json --job_num 136 --pos_enc_dim 136 --log_file $fname
