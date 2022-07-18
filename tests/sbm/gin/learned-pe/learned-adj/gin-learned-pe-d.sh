#!/bin/bash
#$ -N gin-pattern-100k-learned-pe
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GIN_PATTERN_learnedPE_100k.json --job_num ${SGE_TASK_ID} --in_feat_dropout 0.5 --batch_size 4
