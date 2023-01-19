#!/bin/bash
#$ -N GraphTransformer_OGB_ogb_test_20_0_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/20_DEBUG_0_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_OGB_graph_regression.py --config tests/test-configs/GraphTransformer_OGB_OGB_ogb_test.json --job_num 20 --pos_enc_dim 20 --log_file $fname --seed_array 41
