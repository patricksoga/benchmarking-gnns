#!/bin/bash
#$ -N gtn-pattern-500k-adj-dropout-layernorm-3layers
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/pattern_adj_${SGE_TASK_ID}_dropout_layernorm_3layers_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GraphTransformer_PATTERN_PE_500k.json --batch_size 4 --job_num ${SGE_TASK_ID} --log_file $fname --pos_enc_dim 20 --adj_enc --in_feat_dropout 0.4 --L 3 --layer_norm True
