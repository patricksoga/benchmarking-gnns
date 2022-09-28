#!/bin/bash
#$ -N GraphTransformer_CSL_b5-300k-rand-exp1-trials
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 32)
fname=$(pwd)/b5-300k-rand-exp1-trials_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-rand-exp1-trials.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'CSL',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 6,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 5,
#                 'cat_gape': False,
#                 'dataset': 'CSL',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'eigen_bartels_stewart': False,
#                 'experiment_1': True,
#                 'full_graph': False,
#                 'gape_clamp': False,
#                 'gape_individual': False,
#                 'gape_rand': 'none',
#                 'gape_softmax_after': False,
#                 'gape_softmax_before': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'learned_pos_enc': False,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'num_initials': 1,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pos_enc_dim': 20,
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
#  'out_dir': 'out/CSL_graph_classification_b5-300k-rand-exp1-trials',
#  'params': {'batch_size': 5,
#             'epochs': 1000,
#             'init_lr': 0.0005,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 18,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'seed_array': [41, 95, 22, 35],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config test-configs/GraphTransformer_CSL_rand_enc_b5.json --rand_pos_enc False --pos_enc False --rand_pos_enc True --matrix_type A --param_values 32 --job_note b5-300k-rand-exp1-trials --seed_array 41 95 22 35 --experiment_1 True
