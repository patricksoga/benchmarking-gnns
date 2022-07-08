#!/bin/bash
#$ -N gtn-k3c-full
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../
python3 main_K3Colorable_graph_classification.py --config configs/K3Colorable_graph_classification_GraphTransformer_400k_full_graph_BN.json
