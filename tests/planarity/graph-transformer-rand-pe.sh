#!/bin/bash
#$ -N gtn-planarity-500k-learn-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(0 20 40 80 100 120)

conda activate gnn
cd ../../
python3 main_PLANARITY_graph_classification.py --config configs/PLANARITY_graph_classification_GraphTransformer_learnedPE_500k_sparse_graph_BN.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]}
