#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-bnorm-alt-bartels-clamp_128_4_41
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/128_DEBUG_4_41.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-bartels-clamp.json --job_num 128 --pos_enc_dim 128 --log_file $fname --seed_array 41
