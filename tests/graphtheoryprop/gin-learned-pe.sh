#!/bin/bash
#$ -N gin-gtp-100k-learned-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(0 5 10 15 20 30)

conda activate gnn
cd ../../
python3 main_GraphTheoryProp_multitask.py --config configs/GraphTheoryProp_multitask_GIN_GraphTheoryProp_learnedPE_100k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]}
