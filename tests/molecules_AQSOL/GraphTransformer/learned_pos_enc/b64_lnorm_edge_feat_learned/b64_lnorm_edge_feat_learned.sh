#!/bin/bash
#$ -N GraphTransformer_AQSOL_b64_lnorm_edge_feat_learned
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-4:1

pos_enc_dim=(0 4 8 12 16)
fname=$(pwd)/b64_lnorm_edge_feat_learned_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_learned_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_b64_lnorm_edge_feat_learned.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'AQSOL',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 6,
#                 'adj_enc': False,
#                 'batch_norm': False,
#                 'batch_size': 64,
#                 'dataset': 'AQSOL',
#                 'dropout': 0.0,
#                 'edge_feat': True,
#                 'full_graph': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': True,
#                 'learned_pos_enc': True,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'num_train_data': 7000,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pos_enc_dim': 20,
#                 'pow_of_mat': 1,
#                 'rand_pos_enc': False,
#                 'readout': 'sum',
#                 'residual': True,
#                 'self_loop': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/molecules_graph_regression_b64_lnorm_edge_feat_learned',
#  'params': {'batch_size': 64,
#             'epochs': 1000,
#             'init_lr': 0.0005,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config test-configs/GraphTransformer_molecules_b64_bnorm_edge_feat_rand.json --learned_pos_enc True --rand_pos_enc False --init_lr 0.0005 --edge_feat True --job_note b64_lnorm_edge_feat_learned --param_values 4 8 12 16 --varying_param pos_enc_dim --batch_norm False --layer_norm True
