#!/bin/bash
#$ -N gtn-cycles-500k-adj-dropout
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/cycles_adj_${SGE_TASK_ID}_dropout_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GraphTransformer_PE_CYCLES_500k.json --batch_size 2 --job_num ${SGE_TASK_ID} --log_file $fname --batch_size 2 --pos_enc_dim 20 --adj_enc --in_feat_dropout 0.8
