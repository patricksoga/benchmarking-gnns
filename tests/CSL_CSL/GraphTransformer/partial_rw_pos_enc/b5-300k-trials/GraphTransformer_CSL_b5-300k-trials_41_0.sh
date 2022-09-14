#!/bin/bash
#$ -N GraphTransformer_CSL_b5-300k-trials_4_0_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/4_DEBUG_0_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-trials.json --job_num 4 --pos_enc_dim 4 --log_file $fname --seed_array 41
