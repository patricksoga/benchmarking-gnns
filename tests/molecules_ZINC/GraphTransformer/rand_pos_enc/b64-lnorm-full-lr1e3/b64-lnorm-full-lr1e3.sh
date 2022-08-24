#!/bin/bash
#$ -N GraphTransformer_ZINC_b64-lnorm-full-lr1e3
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

pos_enc_dim=(0 4 8 16 32 64)
fname=$(pwd)/b64-lnorm-full-lr1e3_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b64-lnorm-full-lr1e3.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'ZINC',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 6,
#                 'adj_enc': False,
#                 'batch_norm': False,
#                 'batch_size': 64,
#                 'dataset': 'ZINC',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'edge_feat': True,
#                 'full_graph': True,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': True,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'out_dim': 80,
#                 'pos_enc_dim': 8,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'rand_pos_enc': True,
#                 'readout': 'mean',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/molecules_graph_regression_b64-lnorm-full-lr1e3',
#  'params': {'batch_size': 64,
#             'epochs': 1000,
#             'init_lr': 0.001,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 15,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'save_name': 'b64-lnorm-full-lr1e3',
#             'seed': 41,
#             'seed_array': [41],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config ../configs/molecules_graph_regression_GraphTransformer_ZINC_500k.json --batch_size 64 --job_note b64-lnorm-full-lr1e3 --batch_norm False --layer_norm True --rand_pos_enc True --param_values 4 8 16 32 64 --edge_feat True --full_graph True --L 6 --out_dim 80 --hidden_dim 80 --init_lr 0.001 --save_name b64-lnorm-full-lr1e3
