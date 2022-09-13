#!/bin/bash
#$ -N PseudoGraphormer_ZINC_b128-lnorm-500k_1_0_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/1_DEBUG_0_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/PseudoGraphormer_molecules_ZINC_b128-lnorm-500k.json --job_num 1 --pos_enc_dim 1 --log_file $fname --seed_array 41
