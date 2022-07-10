#!/bin/bash
#$ -N gtn-pattern-500k-lape
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GraphTransformer_PATTERN_PE_500k.json
