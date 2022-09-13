#!/bin/bash
#$ -N SAGraphTransformer_CYCLES_b25-noedge_10_1_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/10_DEBUG_1_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/SAGraphTransformer_CYCLES_CYCLES_b25-noedge.json --job_num 10 --pos_enc_dim 10 --log_file $fname --seed_array 41
