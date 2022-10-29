#!/bin/bash
#$ -N GraphTransformer_ZINC_b128-bnorm-alt-noedge-500k-32-stoch-initials15-trials
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 32)
fname=$(pwd)/b128-bnorm-alt-noedge-500k-32-stoch-initials15-trials_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-500k-32-stoch-initials15-trials.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'ZINC',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 128,
#                 'cat_gape': False,
#                 'cycles_k': 6,
#                 'dataset': 'ZINC',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'edge_feat': False,
#                 'eigen_bartels_stewart': False,
#                 'experiment_1': False,
#                 'full_graph': False,
#                 'gape_clamp': False,
#                 'gape_div': False,
#                 'gape_individual': False,
#                 'gape_norm': False,
#                 'gape_normalization': False,
#                 'gape_per_layer': False,
#                 'gape_rand': False,
#                 'gape_scalar': False,
#                 'gape_scale': '0',
#                 'gape_softmax_after': False,
#                 'gape_softmax_before': False,
#                 'gape_squash': 'none',
#                 'gape_stoch': True,
#                 'gape_symmetric': False,
#                 'gape_weight_gen': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'num_initials': 15,
#                 'out_dim': 80,
#                 'pos_enc_dim': 8,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'rand_pos_enc': True,
#                 'rand_sketchy_pos_enc': False,
#                 'random_orientation': False,
#                 'readout': 'mean',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'spectral_attn': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/molecules_graph_regression_b128-bnorm-alt-noedge-500k-32-stoch-initials15-trials',
#  'params': {'batch_size': 128,
#             'epochs': 1000,
#             'init_lr': 0.0007,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 15,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'save_name': 'b128-bnorm-alt-noedge-500k-clamped-8-trials',
#             'seed': 41,
#             'seed_array': [41, 95, 22, 35],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config ../configs/molecules_graph_regression_GraphTransformer_ZINC_500k.json --batch_size 128 --job_note b128-bnorm-alt-noedge-500k-32-stoch-initials15-trials --batch_norm True --layer_norm False --rand_pos_enc True --param_values 32 --seed_array 41 95 22 35 --edge_feat False --init_lr 0.0007 --save_name b128-bnorm-alt-noedge-500k-clamped-8-trials --L 10 --hidden_dim 80 --out_dim 80 --gape_clamp False --gape_stoch True --num_initials 15
