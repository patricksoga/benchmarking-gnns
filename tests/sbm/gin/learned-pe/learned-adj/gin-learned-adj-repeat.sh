#!/bin/bash
#$ -N gin-pattern-100k-learned-pe-repeat-dim
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 10 15)
fname=$(pwd)/repeat_${SGE_TASK_ID}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GIN_PATTERN_learnedPE_100k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --log_file $fname --batch_size 10