#!/bin/bash
#$ -N gatedgcn-pattern-100k-learned-pe-b16
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/repeat_b16_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../../../
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GatedGCN_PATTERN_learnedPE_100k.json --pos_enc_dim 10 --job_num 1 --log_file $fname --batch_size 16
