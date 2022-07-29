#!/bin/bash
#$ -N gtn-graphtheory-500k
#$ -q gpu
#$ -l gpu_card=1

conda activate gnn
cd ../../../../
python3 main_GraphTheoryProp_multitask.py --config configs/GraphTheoryProp_multitask_GraphTransformer_500k.json --in_feat_dropout 0.0 --gin_dropout 0.0 --mlp_dropout 0.0 --embedding_dropout 0.0 --lr 0.1 --batch_size 32 --nb_epochs 100 --nb_runs 1 --random_seed 0 --model_type gtn --nb_classes 2 --data_path data/processed/graphtheoryprop/gtn/ --data_name graphtheoryprop_gtn_baseline_500k --log_dir ./gtn-baseline.log