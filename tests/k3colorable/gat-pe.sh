#!/bin/bash
#$ -N gat-k3c-pe
#$ -q gpu
#$ -l gpu_card=1

source ~/.virtualenvs/hon/bin/activate 
python3 main_K3Colorable_graph_classification.py --config configs/K3Colorable_graph_classification_GAT_PE_100k.json
