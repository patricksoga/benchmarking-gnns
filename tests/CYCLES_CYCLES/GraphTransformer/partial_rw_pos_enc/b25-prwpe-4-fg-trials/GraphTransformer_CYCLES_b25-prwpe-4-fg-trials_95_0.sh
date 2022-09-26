#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-prwpe-4-fg-trials_4_0_95
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/4_DEBUG_0_95.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-prwpe-4-fg-trials.json --job_num 4 --pos_enc_dim 4 --log_file $fname --seed_array 95
