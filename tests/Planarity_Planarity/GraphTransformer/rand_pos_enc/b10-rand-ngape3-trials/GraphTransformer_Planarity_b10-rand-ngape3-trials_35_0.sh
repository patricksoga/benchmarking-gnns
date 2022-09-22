#!/bin/bash
#$ -N GraphTransformer_Planarity_b10-rand-ngape3-trials_16_0_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/16_DEBUG_0_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b10-rand-ngape3-trials.json --job_num 16 --pos_enc_dim 16 --log_file $fname --seed_array 35
