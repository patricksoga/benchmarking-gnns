#!/bin/bash
#$ -N gatedgcn-pattern-100k-lape
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GatedGCN_PATTERN_PE_100k.json
