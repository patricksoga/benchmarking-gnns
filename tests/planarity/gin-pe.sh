#!/bin/bash
#$ -N gin-planarity-pe
#$ -q gpu
#$ -l gpu_card=1

source ~/.virtualenvs/hon/bin/activate 
python3 main_PLANARITY_graph_classification.py --config configs/PLANARITY_graph_classification_GIN_PE_100k.json
