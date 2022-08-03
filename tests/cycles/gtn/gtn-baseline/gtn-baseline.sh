#!/bin/bash
#$ -N gtn-cycles-500k
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/baseline_DEBUG_b25.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GraphTransformer_CYCLES_500k.json --log_file $fname --batch_size 25 --job_num 1
