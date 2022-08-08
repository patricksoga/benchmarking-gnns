#!/bin/bash
#$ -N GIN_SBM_PATTERN_infeatdropout-0.25-b2-random
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3:1

pos_enc_dim=(0 64 128 80)
fname=$(pwd)/infeatdropout-0.25-b2-random_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GIN_SBM_PATTERN_infeatdropout-0.25-b2-random.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'SBM_PATTERN',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GIN',
#  'net_params': {'L': 4,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 2,
#                 'dataset': 'SBM_PATTERN',
#                 'dropout': 0.0,
#                 'gpu_id': 0,
#                 'hidden_dim': 110,
#                 'in_feat_dropout': 0.25,
#                 'learn_eps_GIN': True,
#                 'learned_pos_enc': False,
#                 'matrix_type': 'A',
#                 'n_mlp_GIN': 2,
#                 'neighbor_aggr_GIN': 'sum',
#                 'num_initials': 1,
#                 'pos_enc': False,
#                 'pos_enc_dim': 10,
#                 'pow_of_mat': 1,
#                 'rand_pos_enc': True,
#                 'readout': 'sum',
#                 'residual': True},
#  'out_dir': 'out/SBM_PATTERN_node_classification_infeatdropout-0.25-b2-random',
#  'params': {'batch_size': 2,
#             'epochs': 1000,
#             'init_lr': 0.001,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 5,
#             'max_time': 12,
#             'min_lr': 1e-05,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'weight_decay': 0.0}}

