#!/bin/bash
#$ -N gin-cycles-100k-learned-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(0 5 10 15 20 30)

conda activate gnn
cd ../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GIN_learnedPE_CYCLES_100k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID}
