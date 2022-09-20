#!/bin/bash
#$ -N PseudoGraphormer_CYCLES_b25-500k-trials_1_0_22
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/1_DEBUG_0_22.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/PseudoGraphormer_CYCLES_CYCLES_b25-500k-trials.json --job_num 1 --pos_enc_dim 1 --log_file $fname --seed_array 22
