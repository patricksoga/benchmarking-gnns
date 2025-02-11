#!/bin/bash
#$ -N gtn-cycles-500k-lape
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/lape_DEBUG_b5.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd ../../../../
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GraphTransformer_PE_CYCLES_500k.json --log_file $fname --batch_size 5 --job_num 1
