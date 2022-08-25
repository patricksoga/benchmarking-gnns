#!/bin/bash
#$ -N GraphTransformer_SBM_PATTERN_b26-lnorm-alt-cat_44
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/44_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_PATTERN_b26-lnorm-alt-cat.json --job_num 44 --pos_enc_dim 44 --log_file $fname
