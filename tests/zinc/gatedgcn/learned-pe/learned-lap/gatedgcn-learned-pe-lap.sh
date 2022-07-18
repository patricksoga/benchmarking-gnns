#!/bin/bash
#$ -N gatedgcn-zinc-100k-learned-pe-lap
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 10 20)

conda activate gnn
cd ../../../../
python3 main_molecules_graph_regression.py --config configs/molecules_graph_regression_GatedGCN_ZINC_learnedPE_100k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --edge_feat --matrix_type NL
