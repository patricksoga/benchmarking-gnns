#!/bin/bash
#$ -N gat-planarity-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

models=(GAT GatedGCN GIN GraphTransformer GraphSAGE)
dataset=PLANARITY
model_name=""
params=""
bn=""
full_graph=""
PE=""

conda activate gnn
cd ../../

for model in ${models[@]}; do
    python3 main_${dataset}_graph_classification.py --config configs/${dataset}_graph_classification_${model}_${params}_${bn}_${full_graph}_${PE}.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]}
done