#!/bin/bash
#$ -N GraphTransformer_AQSOL_b128-bnorm-noedge-500k
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3:1

pos_enc_dim=(0 8 16 20)
fname=$(pwd)/b128-bnorm-noedge-500k_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_AQSOL_b128-bnorm-noedge-500k.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'AQSOL',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': False,
#                 'batch_size': 128,
#                 'cat_gape': False,
#                 'dataset': 'AQSOL',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'edge_feat': True,
#                 'eigen_bartels_stewart': False,
#                 'full_graph': False,
#                 'gape_clamp': False,
#                 'gape_individual': False,
#                 'gape_softmax_after': False,
#                 'gape_softmax_before': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': True,
#                 'learned_pos_enc': False,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'num_train_data': 7000,
#                 'out_dim': 80,
#                 'partial_rw_pos_enc': True,
#                 'pos_enc': False,
#                 'pos_enc_dim': 20,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'rand_pos_enc': False,
#                 'rand_sketchy_pos_enc': False,
#                 'random_orientation': False,
#                 'readout': 'mean',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'spectral_attn': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/molecules_graph_regression_b128-bnorm-noedge-500k',
#  'params': {'batch_size': 128,
#             'epochs': 1000,
#             'init_lr': 0.0007,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'save_name': 'b128_bnorm_noedge-500k',
#             'seed': 41,
#             'seed_array': [41],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config test-configs/GraphTransformer_molecules_b64_lnorm_edge_feat_rand.json --varying_param pos_enc_dim --param_values 8 16 20 --batch_size 128 --init_lr 0.0007 --pos_enc False --partial_rw_pos_enc True --rand_pos_enc False --hidden_dim 80 --out_dim 80 --readout mean --L 10 --save_name b128_bnorm_noedge-500k --job_note b128-bnorm-noedge-500k
