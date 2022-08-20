#!/bin/bash
#$ -N GraphTransformer_SBM_PATTERN_b26-lnorm-alt-more_11
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/11_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_PATTERN_b26-lnorm-alt-more.json --job_num 11 --pos_enc_dim 11 --log_file $fname
