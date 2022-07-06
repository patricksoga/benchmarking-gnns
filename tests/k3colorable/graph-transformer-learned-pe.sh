#!/bin/bash
#$ -N gtn-k3c-500k-learned-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(20,40,80,100,120)

conda activate gnn
cd ../../
python3 main_K3Colorable_graph_classification.py --config configs/K3Colorable_graph_classification_GraphTransformer_500k_sparse_graph_BN.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]}
