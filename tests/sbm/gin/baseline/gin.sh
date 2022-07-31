#!/bin/bash
#$ -N gin-pattern-100k
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/baseline.log
touch $fname
fsync -d 10 $fname &
conda activate gnn

cd ../../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GIN_PATTERN_100k.json --log_file $fname
