#!/bin/bash
#$ -N gin-cycles-100k-learned-pe-b25-dout-0.25
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3:1

encdims=(0 10 13 15)
fname=$(pwd)/repeat_b25_dout_0.25_${SGE_TASK_ID}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GIN_learnedPE_CYCLES_100k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --log_file $fname --dropout 0.25
