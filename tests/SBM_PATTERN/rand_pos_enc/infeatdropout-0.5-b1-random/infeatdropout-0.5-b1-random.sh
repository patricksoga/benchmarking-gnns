#!/bin/bash
#$ -N GraphTransformer_SBM_PATTERN_infeatdropout-0.5-b1-random
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3:1

pos_enc_dim=(64 128 80)
fname=$(pwd)/infeatdropout-0.5-b1-random_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.txt
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBM_PATTERN_infeatdropout-0.5-b1-random.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]}

# {'dataset': 'SBM_PATTERN',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 6,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 1,
#                 'dataset': 'SBM_PATTERN',
#                 'dropout': 0.0,
#                 'full_graph': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.5,
#                 'layer_norm': False,
#                 'learned_pos_enc': False,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'num_initials': 1,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pos_enc_dim': 20,
#                 'pow_of_mat': 1,
#                 'rand_pos_enc': True,
#                 'readout': 'sum',
#                 'residual': True,
#                 'self_loop': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/SBM_PATTERN_node_classification_infeatdropout-0.5-b1-random',
#  'params': {'batch_size': 1,
#             'epochs': 1000,
#             'init_lr': 0.0005,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'weight_decay': 0.0}}

