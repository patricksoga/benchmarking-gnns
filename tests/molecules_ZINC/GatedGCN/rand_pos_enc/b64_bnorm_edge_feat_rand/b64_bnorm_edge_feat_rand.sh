#!/bin/bash
#$ -N GatedGCN_ZINC_b64_bnorm_edge_feat_rand
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

pos_enc_dim=(0 8 16 32 64 128)
fname=$(pwd)/b64_bnorm_edge_feat_rand_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GatedGCN_molecules_b64_bnorm_edge_feat_rand.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'ZINC',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GatedGCN',
#  'net_params': {'L': 16,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 64,
#                 'dataset': 'ZINC',
#                 'dropout': 0.0,
#                 'edge_feat': True,
#                 'gpu_id': 0,
#                 'hidden_dim': 78,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'matrix_type': 'A',
#                 'out_dim': 78,
#                 'pos_enc': False,
#                 'pow_of_mat': 1,
#                 'rand_pos_enc': True,
#                 'readout': 'mean',
#                 'residual': True},
#  'out_dir': 'out/molecules_graph_regression_b64_bnorm_edge_feat_rand',
#  'params': {'batch_size': 64,
#             'epochs': 1000,
#             'init_lr': 0.001,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 12,
#             'min_lr': 1e-05,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'weight_decay': 0.0}}

