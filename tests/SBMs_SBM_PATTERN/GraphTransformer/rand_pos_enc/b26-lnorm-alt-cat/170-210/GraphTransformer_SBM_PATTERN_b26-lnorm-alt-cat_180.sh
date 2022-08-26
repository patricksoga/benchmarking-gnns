#!/bin/bash
#$ -N GraphTransformer_SBM_PATTERN_b26-lnorm-alt-cat_180
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/180_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_PATTERN_b26-lnorm-alt-cat.json --job_num 180 --pos_enc_dim 180 --log_file $fname
