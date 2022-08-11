#!/bin/bash
#$ -N GraphTransformer_CSL_lnorm_dim32
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 20)
fname=$(pwd)/lnorm_dim32_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_lnorm_dim32.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'CSL',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 4,
#                 'adj_enc': False,
#                 'batch_norm': False,
#                 'batch_size': 5,
#                 'dataset': 'CSL',
#                 'dropout': 0.0,
#                 'full_graph': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 32,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': True,
#                 'learned_pos_enc': False,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'num_initials': 1,
#                 'out_dim': 32,
#                 'pos_enc': True,
#                 'pos_enc_dim': 20,
#                 'pow_of_mat': 1,
#                 'readout': 'sum',
#                 'residual': True,
#                 'self_loop': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/CSL_graph_classification_lnorm_dim32',
#  'params': {'batch_size': 5,
#             'epochs': 1000,
#             'init_lr': 0.0005,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 18,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'weight_decay': 0.0}}

