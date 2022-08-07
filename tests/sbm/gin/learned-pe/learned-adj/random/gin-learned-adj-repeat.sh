#!/bin/bash
#$ -N gin-pattern-100k-rand-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(0 10 15 20 40 50)
fname=$(pwd)/repeat_${SGE_TASK_ID}_DEBUG_${encdims[${SGE_TASK_ID}]}.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GIN_PATTERN_randPE_100k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --log_file $fname --layer_norm False
