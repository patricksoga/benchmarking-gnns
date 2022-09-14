#!/bin/bash
#$ -N SAGraphTransformer_SBM_PATTERN_b26-500k-sparse-trials
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 10)
fname=$(pwd)/b26-500k-sparse-trials_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/SAGraphTransformer_SBMs_SBM_PATTERN_b26-500k-sparse-trials.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'SBM_PATTERN',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'SAGraphTransformer',
#  'net_params': {'L': 6,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 26,
#                 'cat_gape': False,
#                 'dataset': 'SBM_PATTERN',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'eigen_bartels_stewart': False,
#                 'full_graph': False,
#                 'gape_clamp': False,
#                 'gape_individual': False,
#                 'gape_softmax_after': False,
#                 'gape_softmax_before': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'lpe_dim': 16,
#                 'lpe_layers': 3,
#                 'lpe_n_heads': 4,
#                 'matrix_type': 'A',
#                 'n_heads': 10,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pos_enc_dim': 16,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'rand_pos_enc': False,
#                 'rand_sketchy_pos_enc': False,
#                 'random_orientation': False,
#                 'readout': 'mean',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'spectral_attn': True,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/SBMs_node_classification_b26-500k-sparse-trials',
#  'params': {'batch_size': 26,
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
#python3 configure_tests.py --config ../configs/SBMs_node_clustering_SAGraphTransformer_PATTERN_500k.json --job_note b26-500k-sparse-trials --spectral_attn True --pos_enc False --rand_pos_enc False --param_values 10 --full_graph True --L 6 --full_graph False --batch_size 26 --lpe_dim 16 --seed_array 41 95 22 35
