#!/bin/bash
#$ -N GraphTransformer_ZINC_b64-bnorm-alt-noedge-500k_8_0_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/8_DEBUG_0_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b64-bnorm-alt-noedge-500k.json --job_num 8 --pos_enc_dim 8 --log_file $fname --seed_array 41
