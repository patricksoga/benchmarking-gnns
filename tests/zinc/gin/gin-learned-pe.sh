#!/bin/bash
#$ -N gin-zinc-100k-learned-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(0 5 10 15 20 30)

conda activate gnn
cd ../../../../
python3 main_molecules_graph_regression.py --config configs/molecules_graph_regression_GIN_ZINC_learnedPE_500k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --edge_feat
