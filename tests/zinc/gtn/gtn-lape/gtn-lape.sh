#!/bin/bash
#$ -N gtn-zinc-500k-lape
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../../
python3 main_molecules_graph_regression.py --config configs/molecules_graph_regression_GraphTransformer_ZINC_PE_500k.json --edge_feat
