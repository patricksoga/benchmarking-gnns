#!/bin/bash
#$ -N gtn-cycles-500k-lape
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GraphTransformer_PE_CYCLES_500k.json
