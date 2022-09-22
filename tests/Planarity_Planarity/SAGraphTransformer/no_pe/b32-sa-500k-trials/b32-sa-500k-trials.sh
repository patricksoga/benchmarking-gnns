#!/bin/bash
#$ -N SAGraphTransformer_Planarity_b32-sa-500k-trials
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 10)
fname=$(pwd)/b32-sa-500k-trials_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_Planarity_graph_classification.py --config tests/test-configs/SAGraphTransformer_Planarity_Planarity_b32-sa-500k-trials.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'Planarity',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'SAGraphTransformer',
#  'net_params': {'L': 32,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 32,
#                 'cat_gape': False,
#                 'dataset': 'Planarity',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'edge_feat': False,
#                 'eigen_bartels_stewart': False,
#                 'full_graph': False,
#                 'gape_clamp': False,
#                 'gape_individual': False,
#                 'gape_rand': False,
#                 'gape_softmax_after': False,
#                 'gape_softmax_before': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'lpe_layers': 3,
#                 'lpe_n_heads': 4,
#                 'matrix_type': 'A',
#                 'n_heads': 10,
#                 'num_train_data': 200,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pos_enc_dim': 16,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'rand_sketchy_pos_enc': False,
#                 'random_orientation': False,
#                 'readout': 'mean',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'spectral_attn': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/Planarity_graph_classification_b32-sa-500k-trials',
#  'params': {'batch_size': 32,
#             'epochs': 1000,
#             'init_lr': 0.0005,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'seed_array': [41, 95, 22, 35],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config ../configs/PLANARITY_graph_classification_SAGraphTransformer_500k.json --job_note b32-sa-500k-trials --seed_array 41 95 22 35 --param_values 10
