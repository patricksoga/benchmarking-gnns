#!/bin/bash
#$ -N SAGraphTransformer_CYCLES_b25-noedge-500k-lpe1-lpedim8-vary-trials_20_0_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/20_DEBUG_0_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/SAGraphTransformer_CYCLES_CYCLES_b25-noedge-500k-lpe1-lpedim8-vary-trials.json --job_num 20 --pos_enc_dim 20 --log_file $fname --seed_array 35
