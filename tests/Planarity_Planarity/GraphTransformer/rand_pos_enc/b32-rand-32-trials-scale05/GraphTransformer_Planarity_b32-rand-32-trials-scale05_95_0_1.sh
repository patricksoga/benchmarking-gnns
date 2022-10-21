#!/bin/bash
#$ -N GraphTransformer_Planarity_b32-rand-32-trials-scale05_32_0_95
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/32_DEBUG_0_95.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-rand-32-trials-scale05.json --job_num 32 --pos_enc_dim 32 --log_file $fname --seed_array 95
