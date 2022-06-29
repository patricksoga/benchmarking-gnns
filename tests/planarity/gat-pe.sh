#!/bin/bash
#$ -N gat-planarity-pe
#$ -q gpu
#$ -l gpu_card=1

source ~/.virtualenvs/hon/bin/activate 
python3 main_PLANARITY_graph_classification.py --config configs/PLANARITY_graph_classification_GAT_PE_100k.json
