#!/bin/bash
#$ -N gtn-cycles-500k-learned-pe-stack
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-4:1

initials=(0 2 4 6 8)

fname=$(pwd)/stack_${initials[${SGE_TASK_ID}]}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GraphTransformer_learnedPE_CYCLES_500k.json --pos_enc_dim 15 --job_num ${SGE_TASK_ID} --log_file $fname --in_feat_dropout 0.5 --num_initials ${initials[${SGE_TASK_ID}]}
