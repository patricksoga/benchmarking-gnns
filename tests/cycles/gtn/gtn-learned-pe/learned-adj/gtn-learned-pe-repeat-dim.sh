#!/bin/bash
#$ -N gtn-cycles-500k-learned-pe-dropout-0.25-b25
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 12 13)
fname=$(pwd)/repeat_nodrop_b25_dout_0.25_${SGE_TASK_ID}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GraphTransformer_learnedPE_CYCLES_500k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --log_file $fname --in_feat_dropout 0.25
