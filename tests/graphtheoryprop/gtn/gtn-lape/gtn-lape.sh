#!/bin/bash
#$ -N gtn-graphtheory-500k-lape
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../../
python3 main_GraphTheoryProp_multitask.py --config configs/GraphTheoryProp_multitask_GraphTransformer_PE_500k.json --in_feat_dropout 0.5 --log_dir ./gtn-lape.log