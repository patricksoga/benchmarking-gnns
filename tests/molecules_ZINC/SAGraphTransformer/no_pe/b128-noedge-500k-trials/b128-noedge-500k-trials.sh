#!/bin/bash
#$ -N SAGraphTransformer_ZINC_b128-noedge-500k-trials
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 8)
fname=$(pwd)/b128-noedge-500k-trials_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/SAGraphTransformer_molecules_ZINC_b128-noedge-500k-trials.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'ZINC',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'SAGraphTransformer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 128,
#                 'cat_gape': False,
#                 'dataset': 'ZINC',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'edge_feat': False,
#                 'full_graph': False,
#                 'gape_individual': False,
#                 'gape_softmax_after': False,
#                 'gape_softmax_before': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 72,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'lpe_layers': 3,
#                 'lpe_n_heads': 4,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'out_dim': 72,
#                 'pos_enc_dim': 8,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'random_orientation': False,
#                 'readout': 'sum',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'spectral_attn': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/molecules_graph_regression_b128-noedge-500k-trials',
#  'params': {'batch_size': 128,
#             'epochs': 1000,
#             'init_lr': 0.0007,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'save_name': 'b128-noedge',
#             'seed': 41,
#             'seed_array': [41],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config ../configs/molecules_graph_regression_SAGraphTransformer_ZINC_500k.json --job_note b128-noedge-500k-trials --param_values 8 --save_name b128-noedge --batch_size 128 --edge_feat False --out_dim 72 --hidden_dim 72 --lpe_layers 3
