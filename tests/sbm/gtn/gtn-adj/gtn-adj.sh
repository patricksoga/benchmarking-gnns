#!/bin/bash
#$ -N gtn-pattern-500k-adj
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/pattern_adj_${SGE_TASK_ID}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GraphTransformer_PATTERN_PE_500k.json --batch_size 2 --job_num ${SGE_TASK_ID} --log_file $fname --batch_size 2 --pos_enc_dim 20 --adj_enc
