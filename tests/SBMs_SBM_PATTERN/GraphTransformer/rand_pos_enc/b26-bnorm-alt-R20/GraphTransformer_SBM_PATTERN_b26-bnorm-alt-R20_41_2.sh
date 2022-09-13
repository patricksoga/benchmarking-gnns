#!/bin/bash
#$ -N GraphTransformer_SBM_PATTERN_b26-bnorm-alt-R20_16_2_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/16_DEBUG_2_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_PATTERN_b26-bnorm-alt-R20.json --job_num 16 --pos_enc_dim 16 --log_file $fname --seed_array 41
