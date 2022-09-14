#!/bin/bash
#$ -N GraphTransformer_CSL_b5-300k-trials_16_2_35
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/16_DEBUG_2_35.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-trials.json --job_num 16 --pos_enc_dim 16 --log_file $fname --seed_array 35
