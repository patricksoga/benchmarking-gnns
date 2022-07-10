#!/bin/bash
#$ -N gin-pattern-100k
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GIN_PATTERN_100k.json
