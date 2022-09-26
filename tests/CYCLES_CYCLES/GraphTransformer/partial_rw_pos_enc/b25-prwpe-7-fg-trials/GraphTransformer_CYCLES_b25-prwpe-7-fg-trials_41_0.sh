#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-prwpe-7-fg-trials_7_0_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/7_DEBUG_0_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-prwpe-7-fg-trials.json --job_num 7 --pos_enc_dim 7 --log_file $fname --seed_array 41
