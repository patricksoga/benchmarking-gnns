#!/bin/bash
#$ -N GraphTransformer_ZINC_b32-power_iter
#$ -q gpu@@lalor
#$ -l gpu_card=1

fname=$(pwd)/16_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-1e-3.json --job_num 16 --pos_enc_dim 16 --log_file $fname --power_method True --power_iters 15 --rand_pos_enc False --learned_pos_enc True --batch_size 32 --layer_norm True --batch_norm False
