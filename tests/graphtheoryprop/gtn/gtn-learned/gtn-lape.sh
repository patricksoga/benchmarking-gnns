#!/bin/bash
#$ -N gtn-graphtheory-500k-learned
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-2:1

encdims=(0 15 20)
fname=$(pwd)/repeat_${SGE_TASK_ID}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../
python3 main_GraphTheoryProp_multitask.py --config configs/GraphTheoryProp_multitask_GraphTransformer_learnedPE_500k.json --in_feat_dropout 0.5 --log_dir $fname --job_num ${SGE_TASK_ID} --pos_enc_dim ${encdims[${SGE_TASK_ID}]}