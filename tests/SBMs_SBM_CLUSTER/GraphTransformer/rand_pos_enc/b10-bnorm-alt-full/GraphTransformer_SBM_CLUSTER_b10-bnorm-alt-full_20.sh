#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b10-bnorm-alt-full_20
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/20_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b10-bnorm-alt-full.json --job_num 20 --pos_enc_dim 20 --log_file $fname
