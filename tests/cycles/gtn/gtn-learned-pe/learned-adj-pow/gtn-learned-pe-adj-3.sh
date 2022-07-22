#!/bin/bash
#$ -N gtn-cycles-500k-learned-pe-adj-k3
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 15 20)
touch k3_${SGE_TASK_ID}_DEBUG_.log

conda activate gnn
cd ../../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GraphTransformer_learnedPE_CYCLES_500k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --pow_of_mat 3 > k3_${SGE_TASK_ID}_DEBUG_.log
