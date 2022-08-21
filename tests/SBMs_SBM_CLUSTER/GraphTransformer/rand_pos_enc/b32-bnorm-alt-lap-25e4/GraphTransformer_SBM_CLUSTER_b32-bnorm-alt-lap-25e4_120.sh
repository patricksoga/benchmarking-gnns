#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b32-bnorm-alt-lap-25e4_120
#$ -q gpu@@crc_gpu
#$ -l gpu_card=1

fname=$(pwd)/120_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-bnorm-alt-lap-25e4.json --job_num 120 --pos_enc_dim 120 --log_file $fname
