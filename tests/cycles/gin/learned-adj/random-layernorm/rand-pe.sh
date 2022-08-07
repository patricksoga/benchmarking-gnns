#!/bin/bash
#$ -N gin-cycles-100k-random-pe-lnorm
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(0 32 64 128 256 512)
fname=$(pwd)/rand_${SGE_TASK_ID}_${encdims[${SGE_TASK_ID}]}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns
python3 main_CYCLES_graph_classification.py --config configs/CYCLES_graph_classification_GIN_randPE_CYCLES_100k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --log_file $fname --layer_norm True --batch_norm False
