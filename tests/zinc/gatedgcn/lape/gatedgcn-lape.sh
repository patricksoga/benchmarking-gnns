#!/bin/bash
#$ -N gatedgcn-zinc-100k-lape
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../../
python3 main_molecules_graph_regression.py --config configs/molecules_graph_regression_GatedGCN_ZINC_PE_100k.json
