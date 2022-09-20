#!/bin/bash
#$ -N PseudoGraphormer_SBM_PATTERN_b26-500k-trials_1_0_95
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/1_DEBUG_0_95.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/PseudoGraphormer_SBMs_SBM_PATTERN_b26-500k-trials.json --job_num 1 --pos_enc_dim 1 --log_file $fname --seed_array 95
