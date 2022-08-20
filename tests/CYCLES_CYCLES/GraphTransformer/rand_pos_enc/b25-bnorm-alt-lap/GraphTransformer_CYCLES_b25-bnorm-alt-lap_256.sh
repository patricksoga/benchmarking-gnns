#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-bnorm-alt-lap_256
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/256_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-lap.json --job_num 256 --pos_enc_dim 256 --log_file $fname
