#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b32-bnorm-alt-fg-lr35e5_128_4_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/128_DEBUG_4_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-bnorm-alt-fg-lr35e5.json --job_num 128 --pos_enc_dim 128 --log_file $fname --seed_array 41
