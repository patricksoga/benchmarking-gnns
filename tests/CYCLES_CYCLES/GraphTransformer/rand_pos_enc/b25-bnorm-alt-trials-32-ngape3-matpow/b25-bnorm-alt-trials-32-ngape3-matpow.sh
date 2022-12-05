#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-bnorm-alt-trials-32-ngape3-matpow
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 32)
fname=$(pwd)/b25-bnorm-alt-trials-32-ngape3-matpow_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-trials-32-ngape3-matpow.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'CYCLES',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': False,
#                 'batch_size': 25,
#                 'cat_gape': False,
#                 'cycles_k': 6,
#                 'dataset': 'CYCLES',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'edge_feat': False,
#                 'eigen_bartels_stewart': False,
#                 'experiment_1': False,
#                 'full_graph': False,
#                 'gape_beta': 1.0,
#                 'gape_break_batch': False,
#                 'gape_clamp': False,
#                 'gape_div': False,
#                 'gape_individual': False,
#                 'gape_norm': False,
#                 'gape_normalization': False,
#                 'gape_normalize_mat': False,
#                 'gape_per_layer': False,
#                 'gape_rand': False,
#                 'gape_scalar': False,
#                 'gape_scale': '0',
#                 'gape_softmax_after': False,
#                 'gape_softmax_before': False,
#                 'gape_softmax_init': False,
#                 'gape_squash': 'none',
#                 'gape_stack_strat': '2',
#                 'gape_stoch': False,
#                 'gape_symmetric': False,
#                 'gape_tau': False,
#                 'gape_tau_mat': False,
#                 'gape_uniform_init': False,
#                 'gape_weight_gen': False,
#                 'gape_weight_id': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'learned_pos_enc': False,
#                 'matrix_type': 'A',
#                 'n_gape': 3,
#                 'n_heads': 8,
#                 'ngape_betas': [],
#                 'num_train_data': 200,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pos_enc_dim': 32,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'rand_pos_enc': True,
#                 'rand_sketchy_pos_enc': False,
#                 'random_orientation': False,
#                 'readout': 'sum',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'spectral_attn': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/CYCLES_graph_classification_b25-bnorm-alt-trials-32-ngape3-matpow',
#  'params': {'batch_size': 25,
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
#python3 configure_tests.py --config ../configs/CYCLES_graph_classification_GraphTransformer_CYCLES_500k.json --batch_size 25 --job_note b25-bnorm-alt-trials-32-ngape3-matpow --rand_pos_enc True --seed_array 41 95 22 35 --gape_clamp False --batch_norm False --layer_norm False --edge_feat False --full_graph False --n_gape 3 --pos_enc_dim 32 --param_values 32
