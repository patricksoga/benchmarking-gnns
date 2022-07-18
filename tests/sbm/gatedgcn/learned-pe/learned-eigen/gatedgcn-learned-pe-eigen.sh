#!/bin/bash
#$ -N gatedgcn-pattern-100k-learned-pe-eigen
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 10 20)

conda activate gnn
cd ../../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GatedGCN_PATTERN_learnedPE_100k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --matrix_type E
