#!/bin/bash
#$ -N GraphTransformer_AQSOL_b64_bnorm_edge_feat_rand_dout
#$ -q *@@lalor
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 16)
fname=$(pwd)/b64_bnorm_edge_feat_rand_dout_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_b64_bnorm_edge_feat_rand_dout.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'AQSOL',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 6,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 64,
#                 'dataset': 'AQSOL',
#                 'dropout': 0.35,
#                 'edge_feat': True,
#                 'full_graph': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'learned_pos_enc': False,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'num_train_data': 7000,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pos_enc_dim': 20,
#                 'pow_of_mat': 1,
#                 'rand_pos_enc': True,
#                 'readout': 'sum',
#                 'residual': True,
#                 'self_loop': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/molecules_graph_regression_b64_bnorm_edge_feat_rand_dout',
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
#python3 configure_tests.py --config test-configs/GraphTransformer_molecules_b64_bnorm_edge_feat_rand.json --init_lr 0.0005 --edge_feat True --job_note b64_bnorm_edge_feat_rand_dout --dropout 0.35 --param_values 16 --varying_param pos_enc_dim
