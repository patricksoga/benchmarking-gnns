#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-alt-noedge-clamp-ngape4-4-ind-trials_4_0_22
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/4_DEBUG_0_22.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-alt-noedge-clamp-ngape4-4-ind-trials.json --job_num 4 --pos_enc_dim 4 --log_file $fname --seed_array 22
