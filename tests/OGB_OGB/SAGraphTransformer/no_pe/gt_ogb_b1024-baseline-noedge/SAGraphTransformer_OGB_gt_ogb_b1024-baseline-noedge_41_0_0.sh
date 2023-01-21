#!/bin/bash
#$ -N SAGraphTransformer_OGB_gt_ogb_b1024-baseline-noedge_10_0_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/10_DEBUG_0_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_OGB_graph_regression.py --config tests/test-configs/SAGraphTransformer_OGB_OGB_gt_ogb_b1024-baseline-noedge.json --job_num 10 --pos_enc_dim 10 --log_file $fname --seed_array 41
