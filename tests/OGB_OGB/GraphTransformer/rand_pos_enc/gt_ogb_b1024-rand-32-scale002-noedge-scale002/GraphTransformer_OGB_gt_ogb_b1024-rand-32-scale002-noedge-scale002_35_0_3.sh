#!/bin/bash
#$ -N GraphTransformer_OGB_gt_ogb_b1024-rand-32-scale002-noedge-scale002_32_0_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/32_DEBUG_0_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_OGB_graph_regression.py --config tests/test-configs/GraphTransformer_OGB_OGB_gt_ogb_b1024-rand-32-scale002-noedge-scale002.json --job_num 32 --pos_enc_dim 32 --log_file $fname --seed_array 35
