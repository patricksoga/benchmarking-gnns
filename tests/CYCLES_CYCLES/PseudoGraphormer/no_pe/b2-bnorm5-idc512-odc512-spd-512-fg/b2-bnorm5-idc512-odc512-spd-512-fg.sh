#!/bin/bash
#$ -N PseudoGraphormer_CYCLES_b2-bnorm5-idc512-odc512-spd-512-fg
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 1)
fname=$(pwd)/b2-bnorm5-idc512-odc512-spd-512-fg_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/PseudoGraphormer_CYCLES_CYCLES_b2-bnorm5-idc512-odc512-spd-512-fg.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'CYCLES',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'PseudoGraphormer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 25,
#                 'cat_gape': False,
#                 'dataset': 'CYCLES',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'full_graph': True,
#                 'gape_individual': False,
#                 'gape_softmax_after': False,
#                 'gape_softmax_before': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_deg_centrality': 512,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'num_train_data': 200,
#                 'out_deg_centrality': 512,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'readout': 'mean',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'spd_len': 512,
#                 'spectral_attn': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/CYCLES_graph_classification_b2-bnorm5-idc512-odc512-spd-512-fg',
#  'params': {'batch_size': 25,
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
#python3 configure_tests.py --config ../configs/CYCLES_PseudoGraphormer_500k.json --job_note b2-bnorm5-idc512-odc512-spd-512-fg --batch_size 25 --param_values 1 --batch_norm True --layer_norm False --full_graph True
