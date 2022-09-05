#!/bin/bash
#$ -N GraphTransformer_SBM_PATTERN_b26-bnorm-alt-ngape4-softmax
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

pos_enc_dim=(0 8 16 32 64 128)
fname=$(pwd)/b26-bnorm-alt-ngape4-softmax_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_PATTERN_b26-bnorm-alt-ngape4-softmax.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'SBM_PATTERN',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 26,
#                 'cat_gape': False,
#                 'dataset': 'SBM_PATTERN',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'full_graph': False,
#                 'gape_individual': False,
#                 'gape_softmax_after': True,
#                 'gape_softmax_before': True,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'matrix_type': 'A',
#                 'n_gape': 4,
#                 'n_heads': 8,
#                 'out_dim': 80,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'rand_pos_enc': True,
#                 'readout': 'mean',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'spectral_attn': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/SBMs_node_classification_b26-bnorm-alt-ngape4-softmax',
#  'params': {'batch_size': 26,
#             'epochs': 1000,
#             'init_lr': 0.0005,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'seed_array': [41],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config ../configs/SBMs_node_clustering_GraphTransformer_PATTERN_500k.json --job_note b26-bnorm-alt-ngape4-softmax --param_values 8 16 32 64 128 --batch_size 26 --rand_pos_enc True --batch_norm True --layer_norm False --n_gape 4 --gape_individual False --gape_softmax_after True --gape_softmax_before True
