#!/bin/bash
#$ -N GraphTransformer_ZINC_b128-bnorm-alt-1e-3-lap_64
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/64_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-1e-3-lap.json --job_num 64 --pos_enc_dim 64 --log_file $fname
