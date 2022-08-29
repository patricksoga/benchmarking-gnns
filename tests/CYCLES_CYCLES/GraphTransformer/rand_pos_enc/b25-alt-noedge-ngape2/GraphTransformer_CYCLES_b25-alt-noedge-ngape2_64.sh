#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-alt-noedge-ngape2_64
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/64_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-alt-noedge-ngape2.json --job_num 64 --pos_enc_dim 64 --log_file $fname
