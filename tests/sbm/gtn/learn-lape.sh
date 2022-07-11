#!/bin/bash
#$ -N gtn-pattern-500k-learned-lape-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 20 30)

conda activate gnn
cd ../../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GraphTransformer_PATTERN_learnedlapePE_500k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --batch_size 4
