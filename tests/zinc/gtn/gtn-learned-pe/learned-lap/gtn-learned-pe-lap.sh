#!/bin/bash
#$ -N gtn-zinc-500k-learned-pe-lap
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3:1

encdims=(0 5 10 20)

conda activate gnn
cd ../../../../
python3 main_molecules_graph_regression.py --config configs/molecules_graph_regression_GraphTransformer_ZINC_learnedPE_500k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --edge_feat --matrix_type NL
