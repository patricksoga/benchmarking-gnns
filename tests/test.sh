#!/bin/bash
#$ -N GraphTransformer_CYCLES_batch_norm
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3:1

L=(3 4 5)
fname=$(pwd)/batch_norm_${SGE_TASK_ID}_${L[${SGE_TASK_ID}]}_DEBUG.txt
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_batch_norm.json --job_num ${SGE_TASK_ID} --L ${L[${SGE_TASK_ID}]}

# {'dataset': 'CYCLES',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 6,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 10,
#                 'dataset': 'CYCLES',
#                 'dropout': 0.0,
#                 'full_graph': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'learned_pos_enc': False,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'num_initials': 1,
#                 'num_train_data': 200,
#                 'out_dim': 80,
#                 'pos_enc': True,
#                 'pos_enc_dim': 20,
#                 'pow_of_mat': 1,
#                 'readout': 'sum',
#                 'residual': True,
#                 'self_loop': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/CYCLES_graph_classification_batch_norm',
#  'params': {'batch_size': 10,
#             'epochs': 1000,
#             'init_lr': 0.0005,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'weight_decay': 0.0}}

