#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-bnorm-edge_180
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/180_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-edge.json --job_num 180 --pos_enc_dim 180 --log_file $fname
