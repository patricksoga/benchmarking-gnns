#!/bin/bash
#$ -N GraphTransformer_CYCLES
#$ -q gpu@@lalor
#$ -l gpu_card=1

fname1=$(pwd)/edge_feat.log
fname2=$(pwd)/b25.log

touch $fname1
touch $fname2

fsync -d 10 $fname1 &
fsync -d 10 $fname2

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config test-configs/GraphTransformer_CYCLES_CYCLES_b5-lnorm-alt-edge_feat.json --job_num 1 --pos_enc_dim 128 --log_file $fname1 &
python3 main_CYCLES_graph_classification.py --config test-configs/GraphTransformer_CYCLES_CYCLES-b25-lnorm-alt.json --job_num 2 --pos_enc_dim 64 --log_file $fname2

