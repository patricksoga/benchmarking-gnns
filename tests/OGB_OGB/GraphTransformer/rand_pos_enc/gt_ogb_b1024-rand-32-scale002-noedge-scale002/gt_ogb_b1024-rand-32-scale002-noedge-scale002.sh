#!/bin/bash
#$ -N GraphTransformer_OGB_gt_ogb_b1024-rand-32-scale002-noedge-scale002
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 32)
fname=$(pwd)/gt_ogb_b1024-rand-32-scale002-noedge-scale002_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_OGB_graph_regression.py --config tests/test-configs/GraphTransformer_OGB_OGB_gt_ogb_b1024-rand-32-scale002-noedge-scale002.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'OGB',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 1024,
#                 'cat_gape': False,
#                 'cycles_k': 6,
#                 'dataset': 'OGB',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'edge_feat': False,
#                 'eigen_bartels_stewart': False,
#                 'experiment_1': False,
#                 'full_graph': False,
#                 'gape_beta': 1.0,
#                 'gape_break_batch': False,
#                 'gape_clamp': False,
#                 'gape_cond_lbl': False,
#                 'gape_div': False,
#                 'gape_individual': False,
#                 'gape_norm': False,
#                 'gape_normalization': False,
#                 'gape_normalize_mat': False,
#                 'gape_per_layer': False,
#                 'gape_rand': False,
#                 'gape_scalar': False,
#                 'gape_scale': '0.02',
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
#                 'n_heads': 8,
#                 'ngape_agg': 'sum',
#                 'ngape_betas': [],
#                 'out_dim': 80,
#                 'partial_rw_pos_enc': False,
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
#  'out_dir': 'out/OGB_graph_regression_gt_ogb_b1024-rand-32-scale002-noedge-scale002',
#  'params': {'batch_size': 1024,
#             'epochs': 1000,
#             'init_lr': 0.0007,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 15,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'save_name': 'b128-prwpe',
#             'seed': 41,
#             'seed_array': [41, 95, 22, 35],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --param_values 32 --job_note gt_ogb_b1024-rand-32-scale002-noedge --batch_size 1024 --learned_pos_enc False --layer_norm False --batch_norm True --partial_rw_pos_enc False --dataset OGB --config ./test-configs/GraphTransformer_molecules_ZINC_b128-prwpe-noedge.json --hidden_dim 80 --out_dim 80 --seed_array 41 95 22 35 --rand_pos_enc True --gape_scale 0.02
