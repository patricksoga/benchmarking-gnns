#!/bin/bash
#$ -N gtn-planarity-pe
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../
python3 main_PLANARITY_graph_classification.py --config configs/PLANARITY_graph_classification_GraphTransformer_PE_400k_sparse_graph_BN.json

