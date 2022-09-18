#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-alt-noedge-clamp-ngape6-2-trials_2_0_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/2_DEBUG_0_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-alt-noedge-clamp-ngape6-2-trials.json --job_num 2 --pos_enc_dim 2 --log_file $fname --seed_array 35
