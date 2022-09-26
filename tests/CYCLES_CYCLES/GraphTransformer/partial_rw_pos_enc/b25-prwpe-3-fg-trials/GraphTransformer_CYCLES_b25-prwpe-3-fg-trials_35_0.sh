#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-prwpe-3-fg-trials_3_0_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/3_DEBUG_0_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-prwpe-3-fg-trials.json --job_num 3 --pos_enc_dim 3 --log_file $fname --seed_array 35
