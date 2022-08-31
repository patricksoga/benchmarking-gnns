#!/bin/bash
#$ -N PseudoGraphormer_CYCLES_b2-bnorm5-idc512-odc512-spd-512_1
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/1_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/PseudoGraphormer_CYCLES_CYCLES_b2-bnorm5-idc512-odc512-spd-512.json --job_num 1 --pos_enc_dim 1 --log_file $fname
