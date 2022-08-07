#!/bin/bash
#$ -N gtn-pattern-500k-rand-pe
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

encdims=(0 32 64 128 256 512)
fname=$(pwd)/rand_layernorm_${SGE_TASK_ID}_${encdims[${SGE_TASK_ID}]}_DEBUG_.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns
python3 main_SBMs_node_classification.py --config configs/SBMs_node_clustering_GraphTransformer_PATTERN_randPE_500k.json --pos_enc_dim ${encdims[${SGE_TASK_ID}]} --job_num ${SGE_TASK_ID} --batch_size 1 --log_file $fname --layer_norm True --batch_norm False
