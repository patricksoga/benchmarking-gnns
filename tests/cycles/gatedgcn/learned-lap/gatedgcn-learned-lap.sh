#!/bin/bash
#$ -N gatedgcn-cycles-100k-learned-pe-laplacian
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 10 15)
fname=$(pwd)/laplacian_${SGE_TASK_ID}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GatedGCN_learnedPE_CYCLES_100k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --log_file $fname --batch_size 10 --matrix_type L
