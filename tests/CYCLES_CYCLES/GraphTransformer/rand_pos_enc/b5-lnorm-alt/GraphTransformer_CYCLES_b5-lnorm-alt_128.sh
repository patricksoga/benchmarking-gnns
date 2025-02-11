#!/bin/bash
#$ -N GraphTransformer_CYCLES_b5-lnorm-alt
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/128_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_b5-lnorm-alt.json --job_num 128 --pos_enc_dim 128 --log_file $fname
