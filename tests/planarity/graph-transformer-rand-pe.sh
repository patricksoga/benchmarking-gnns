#!/bin/bash
#$ -N gtn-planarity-500k-rand-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(0 5 10 15 20 30)

conda activate gnn
cd ../../
python3 main_PLANARITY_graph_classification.py --config configs/PLANARITY_graph_classification_GraphTransformer_randPE_500k_sparse_graph_BN.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]}
