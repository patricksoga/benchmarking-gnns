#!/bin/bash
#$ -N GraphTransformer_SBM_CLUSTER_b32-lnorm-alt-32-scale40-trials
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 32)
fname=$(pwd)/b32-lnorm-alt-32-scale40-trials_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-lnorm-alt-32-scale40-trials.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'SBM_CLUSTER',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': False,
#                 'batch_size': 32,
#                 'cat_gape': False,
#                 'cycles_k': 6,
#                 'dataset': 'SBM_CLUSTER',
#                 'diag': False,
#                 'dropout': 0.0,
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
#                 'gape_scale': '0.025',
#                 'gape_softmax_after': False,
#                 'gape_softmax_before': False,
#                 'gape_squash': 'none',
#                 'gape_symmetric': False,
#                 'gape_weight_gen': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': True,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'out_dim': 80,
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
#  'out_dir': 'out/SBMs_node_classification_b32-lnorm-alt-32-scale40-trials',
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
#python3 configure_tests.py --config ../configs/SBMs_node_clustering_GraphTransformer_CLUSTER_500k.json --job_note b32-lnorm-alt-32-scale40-trials --param_values 32 --batch_size 32 --seed_array 41 95 22 35 --rand_pos_enc True --batch_norm False --layer_norm True --gape_clamp False --gape_scale 0.025
