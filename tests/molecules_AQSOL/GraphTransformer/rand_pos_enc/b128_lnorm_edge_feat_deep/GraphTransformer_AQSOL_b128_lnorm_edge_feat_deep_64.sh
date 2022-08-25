#!/bin/bash
#$ -N GraphTransformer_AQSOL_b128_lnorm_edge_feat_deep_64
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/64_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_AQSOL_b128_lnorm_edge_feat_deep.json --job_num 64 --pos_enc_dim 64 --log_file $fname
