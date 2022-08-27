#!/bin/bash
#$ -N SAGraphTransformer_SBM_CLUSTER_b16-sparse_32
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/32_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/SAGraphTransformer_SBMs_SBM_CLUSTER_b16-sparse.json --job_num 32 --pos_enc_dim 32 --log_file $fname
