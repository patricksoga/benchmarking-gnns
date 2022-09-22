#!/bin/bash
#$ -N GraphTransformer_Planarity_b10-lape-20--trials_20_0_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/20_DEBUG_0_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b10-lape-20--trials.json --job_num 20 --pos_enc_dim 20 --log_file $fname --seed_array 35
