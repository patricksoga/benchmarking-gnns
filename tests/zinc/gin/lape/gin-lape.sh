#!/bin/bash
#$ -N gin-zinc-100k-lape
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../../
python3 main_molecules_graph_regression.py --config configs/molecules_graph_regression_GIN_ZINC_PE_100k.json
