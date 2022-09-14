#!/bin/bash
#$ -N GraphTransformer_CSL_b5-300k-trials_32_4_22
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/32_DEBUG_4_22.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-trials.json --job_num 32 --pos_enc_dim 32 --log_file $fname --seed_array 22
