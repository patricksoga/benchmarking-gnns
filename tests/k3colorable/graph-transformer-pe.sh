#!/bin/bash
#$ -N gtn-k3c-pe
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../
python3 main_K3Colorable_graph_classification.py --config configs/K3Colorable_graph_classification_GraphTransformer_PE_400k_sparse_graph_BN.json

