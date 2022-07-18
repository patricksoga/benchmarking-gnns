#!/bin/bash
#$ -N gtn-planarity-500k-learn-pe-eigen
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 15 20)

conda activate gnn
cd ../../../../
python3 main_PLANARITY_graph_classification.py --config configs/PLANARITY_graph_classification_GraphTransformer_learnedPE_500k_sparse_graph_BN.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --matrix_type E
