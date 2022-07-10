#!/bin/bash
#$ -N gtn-cycles-500k
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GraphTransformer_CYCLES_500k.json
