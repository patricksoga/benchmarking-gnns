#!/bin/bash
#$ -N GraphTransformer_ZINC_b128-bnorm-alt-noedge-R_44
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/44_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-R.json --job_num 44 --pos_enc_dim 44 --log_file $fname
