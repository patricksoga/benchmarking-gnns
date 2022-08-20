#!/bin/bash
#$ -N GraphTransformer_SBM_PATTERN_b26-lnorm-alt-lap_7
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/7_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_PATTERN_b26-lnorm-alt-lap.json --job_num 7 --pos_enc_dim 7 --log_file $fname
