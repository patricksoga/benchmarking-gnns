#!/bin/bash
#$ -N gin-planarity
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../
python3 main_PLANARITY_graph_classification.py --config configs/PLANARITY_graph_classification_GIN_100k.json
