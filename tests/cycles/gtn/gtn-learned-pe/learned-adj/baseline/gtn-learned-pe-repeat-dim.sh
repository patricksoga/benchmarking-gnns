#!/bin/bash
#$ -N gtn-cycles-500k-learned-pe-repeat-dim
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(0 10 11 12 13 14 15)
fname=$(pwd)/repeat_${SGE_TASK_ID}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GraphTransformer_learnedPE_CYCLES_500k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --log_file $fname
