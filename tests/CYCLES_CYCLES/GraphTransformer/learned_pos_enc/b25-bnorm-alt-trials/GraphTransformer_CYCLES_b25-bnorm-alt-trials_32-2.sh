#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-bnorm-alt-trials_32-2
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/32_DEBUG_2.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-trials.json --job_num 2 --pos_enc_dim 16 --log_file $fname
