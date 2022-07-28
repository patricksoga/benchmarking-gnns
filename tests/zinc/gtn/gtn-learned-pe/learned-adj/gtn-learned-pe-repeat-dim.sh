#!/bin/bash
#$ -N gtn-zinc-500k-learned-pe-repeat-dim
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 15 20)
fname=$(pwd)/repeat_${SGE_TASK_ID}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../../
python3 main_molecules_graph_regression.py --config configs/molecules_graph_regression_GraphTransformer_ZINC_learnedPE_500k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --batch_size 4 --log_file $fname --in_feat_dropout 0.5
