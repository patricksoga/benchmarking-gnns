#!/bin/bash
#$ -N gcn-planarity-pe
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../
python3 main_PLANARITY_graph_classification.py --config configs/PLANARITY_graph_classification_GatedGCN_PE_100k.json
