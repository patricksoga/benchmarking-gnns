#!/bin/bash
#$ -N gcn-k3c
#$ -q gpu
#$ -l gpu_card=1

source ~/.virtualenvs/hon/bin/activate 
python3 main_K3Colorable_graph_classification.py --config configs/K3Colorable_graph_classification_GatedGCN_100k.json
