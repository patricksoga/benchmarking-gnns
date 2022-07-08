#!/bin/bash
#$ -N gin-k3c-pe
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../
python3 main_K3Colorable_graph_classification.py --config configs/K3Colorable_graph_classification_GIN_PE_100k.json
