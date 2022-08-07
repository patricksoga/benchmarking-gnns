#!/bin/bash
#$ -N GraphTransformer_CYCLES_Hello
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3:1

thingy=(3 4 5)
fname=$(pwd)/Hello_${SGE_TASK_ID}_${args.varying_param[${SGE_TASK_ID}]}_DEBUG.txt
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

